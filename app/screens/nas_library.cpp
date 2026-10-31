// NASBridge - Reads the games on the NAS, in the background, through a cache.
// Copyright (C) 2026 NASBridge contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "concepts/nas_library.hpp"

#include "concepts/library_cache.hpp"
#include "concepts/nas.hpp"
#include "concepts/nas_probe.hpp"

#include <algorithm>
#include <cstring>
#include <atomic>
#include <deque>
#include <map>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#if defined(__PROSPERO__)
#include <pthread_np.h>
#include <sys/cpuset.h>
extern "C" int sceKernelGetCurrentCpu(void);
#endif

namespace hui::concepts
{

namespace
{

constexpr std::size_t kMaxParam = 256u * 1024u;
constexpr std::size_t kMaxIcon = 16u * 1024u * 1024u; // pic0.png can stand in
constexpr int kCoverSide = 512; // covers are kept at most this big

void yield_now()
{
    sched_yield();
}

// The UI thread's core, so the reader can keep off it. Console threads of one
// priority are never time-sliced: a cover decoding on the UI's core would
// hold up a frame until it finished.
int current_cpu()
{
#if defined(__PROSPERO__)
    return sceKernelGetCurrentCpu();
#else
    return -1;
#endif
}

// Moves the calling thread off `avoid` (best effort: on failure it stays put).
void keep_off_cpu(int avoid)
{
#if defined(__PROSPERO__)
    if (avoid < 0)
        return;
    cpuset_t allowed;
    CPU_ZERO(&allowed);
    if (pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed) != 0)
        return;
    if (!CPU_ISSET(avoid, &allowed))
        return;
    CPU_CLR(avoid, &allowed);
    if (CPU_EMPTY(&allowed))
        return;
    pthread_setaffinity_np(pthread_self(), sizeof(allowed), &allowed);
#else
    (void)avoid;
#endif
}

// ---- a little JSON: string values by key ----

void append_utf8(std::string *out, std::uint32_t code)
{
    if (code < 0x80)
    {
        *out += static_cast<char>(code);
    }
    else if (code < 0x800)
    {
        *out += static_cast<char>(0xc0 | (code >> 6));
        *out += static_cast<char>(0x80 | (code & 0x3f));
    }
    else if (code < 0x10000)
    {
        *out += static_cast<char>(0xe0 | (code >> 12));
        *out += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
        *out += static_cast<char>(0x80 | (code & 0x3f));
    }
    else
    {
        *out += static_cast<char>(0xf0 | (code >> 18));
        *out += static_cast<char>(0x80 | ((code >> 12) & 0x3f));
        *out += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
        *out += static_cast<char>(0x80 | (code & 0x3f));
    }
}

int hex4(const std::string &text, std::size_t at)
{
    if (at + 4 > text.size())
        return -1;
    int value = 0;
    for (std::size_t i = at; i < at + 4; ++i)
    {
        const char c = text[i];
        int digit = -1;
        if (c >= '0' && c <= '9')
            digit = c - '0';
        else if (c >= 'a' && c <= 'f')
            digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F')
            digit = c - 'A' + 10;
        if (digit < 0)
            return -1;
        value = value * 16 + digit;
    }
    return value;
}

// Reads the JSON string starting at the quote at `at`; returns where it ends.
std::size_t read_string(const std::string &text, std::size_t at, std::string *out)
{
    out->clear();
    if (at >= text.size() || text[at] != '"')
        return std::string::npos;
    for (std::size_t i = at + 1; i < text.size(); ++i)
    {
        const char c = text[i];
        if (c == '"')
            return i + 1;
        if (c != '\\')
        {
            *out += c;
            continue;
        }
        if (++i >= text.size())
            break;
        switch (text[i])
        {
        case 'n':
            *out += '\n';
            break;
        case 't':
            *out += '\t';
            break;
        case 'r':
        case 'b':
        case 'f':
            break;
        case 'u':
        {
            int code = hex4(text, i + 1);
            if (code < 0)
                return std::string::npos;
            i += 4;
            if (code >= 0xd800 && code < 0xdc00 && i + 6 < text.size() && text[i + 1] == '\\' &&
                text[i + 2] == 'u')
            {
                const int low = hex4(text, i + 3);
                if (low >= 0xdc00 && low < 0xe000)
                {
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                    i += 6;
                }
            }
            append_utf8(out, static_cast<std::uint32_t>(code));
            break;
        }
        default:
            *out += text[i]; // \" \\ \/
            break;
        }
    }
    return std::string::npos;
}

// The string value of the first "key" at or after `from`; "" when absent.
std::string string_value(const std::string &json, const char *key, std::size_t from = 0,
                         std::size_t *found_at = nullptr)
{
    const std::string quoted = std::string("\"") + key + "\"";
    std::size_t at = json.find(quoted, from);
    while (at != std::string::npos)
    {
        std::size_t p = at + quoted.size();
        while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\r' ||
                                   json[p] == '\n'))
            ++p;
        if (p < json.size() && json[p] == ':')
        {
            ++p;
            while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\r' ||
                                       json[p] == '\n'))
                ++p;
            std::string value;
            if (read_string(json, p, &value) != std::string::npos)
            {
                if (found_at != nullptr)
                    *found_at = at;
                return value;
            }
            return "";
        }
        at = json.find(quoted, at + 1);
    }
    return "";
}

std::string without_extension(const std::string &name)
{
    const std::size_t dot = name.rfind('.');
    return dot == std::string::npos || dot == 0 ? name : name.substr(0, dot);
}

// Keeps the middle square of a wide (or tall) picture.
void crop_square(Image *image)
{
    const int side = std::min(image->width, image->height);
    const int x0 = (image->width - side) / 2;
    const int y0 = (image->height - side) / 2;
    std::vector<std::uint8_t> square(static_cast<std::size_t>(side) * side * 4);
    for (int y = 0; y < side; ++y)
        std::memcpy(square.data() + static_cast<std::size_t>(y) * side * 4,
                    image->rgba.data() +
                        (static_cast<std::size_t>(y0 + y) * image->width + x0) * 4,
                    static_cast<std::size_t>(side) * 4);
    image->width = side;
    image->height = side;
    image->rgba = std::move(square);
}

bool title_before(const NasGame &a, const NasGame &b)
{
    const std::string x = lowered(a.title);
    const std::string y = lowered(b.title);
    return x != y ? x < y : a.entry < b.entry;
}

} // namespace

void read_param_json(const std::string &json, NasGame *game)
{
    const std::string id = string_value(json, "titleId");
    if (!id.empty())
        game->title_id = id;
    const std::string version = string_value(json, "contentVersion");
    if (!version.empty())
        game->version = version;

    // "localizedParameters": { "defaultLanguage": "en-US", "en-US": { "titleName": ... } }
    std::string title;
    const std::size_t localized = json.find("\"localizedParameters\"");
    if (localized != std::string::npos)
    {
        const std::string language = string_value(json, "defaultLanguage", localized);
        if (!language.empty())
        {
            const std::string quoted = "\"" + language + "\"";
            std::size_t at = json.find(quoted, localized);
            // Skip the "defaultLanguage": "en-US" value itself.
            while (at != std::string::npos)
            {
                std::size_t p = at + quoted.size();
                while (p < json.size() && (json[p] == ' ' || json[p] == '\t'))
                    ++p;
                if (p < json.size() && json[p] == ':')
                    break;
                at = json.find(quoted, at + 1);
            }
            if (at != std::string::npos)
                title = string_value(json, "titleName", at);
        }
    }
    if (title.empty())
        title = string_value(json, "titleName");
    if (!title.empty())
        game->title = title;
}

void cover_tones(const Image &image, float tone[3][3])
{
    double sum[3] = {0, 0, 0};
    int count = 0;
    float best = -1.0f;
    float accent[3] = {0.5f, 0.5f, 0.5f};
    const int step_x = image.width > 16 ? image.width / 16 : 1;
    const int step_y = image.height > 16 ? image.height / 16 : 1;
    for (int y = step_y / 2; y < image.height; y += step_y)
        for (int x = step_x / 2; x < image.width; x += step_x)
        {
            const std::uint8_t *p =
                image.rgba.data() + (static_cast<std::size_t>(y) * image.width + x) * 4;
            const float r = p[0] / 255.0f;
            const float g = p[1] / 255.0f;
            const float b = p[2] / 255.0f;
            sum[0] += r;
            sum[1] += g;
            sum[2] += b;
            ++count;
            const float high = std::max(r, std::max(g, b));
            const float low = std::min(r, std::min(g, b));
            const float vivid = (high - low) * high; // saturation, weighted by brightness
            if (vivid > best)
            {
                best = vivid;
                accent[0] = r;
                accent[1] = g;
                accent[2] = b;
            }
        }
    for (int c = 0; c < 3; ++c)
    {
        const float mid = count > 0 ? static_cast<float>(sum[c] / count) : 0.3f;
        tone[1][c] = mid;
        tone[0][c] = mid * 0.35f;
        tone[2][c] = accent[c];
    }
}

// ---- the job ----

namespace
{

constexpr int kMaxQueuedPictures = 3; // decoded pictures waiting for the UI, at most
constexpr int kDepth = 3;             // bounded search below a game folder for sce_sys
constexpr int kListings = 10;         // ... in at most this many listings
constexpr int kSizeFolders = 500;     // hard cap: don't traverse huge NAS trees for a size

// What the NAS says about a file or folder, to notice when it changes.
std::string stamp_of(const DavEntry &entry)
{
    return std::to_string(entry.size) + "|" + entry.modified + "|" + entry.etag;
}

// .zip, .rar, .7z, and split parts: name.7z.001, name.zip.001, name.r00,
// name.z01, name.part2.rar (ends in .rar anyway).
bool is_archive(const std::string &name)
{
    const std::string low = lowered(name);
    if (ends_with_nocase(low, ".zip") || ends_with_nocase(low, ".rar") ||
        ends_with_nocase(low, ".7z"))
        return true;
    const std::size_t dot = low.rfind('.');
    if (dot == std::string::npos || dot + 1 >= low.size())
        return false;
    const std::string ext = low.substr(dot + 1);
    const bool digits = ext.find_first_not_of("0123456789") == std::string::npos;
    if (digits) // name.7z.001, name.zip.001
    {
        const std::string before = low.substr(0, dot);
        return ends_with_nocase(before, ".7z") || ends_with_nocase(before, ".zip") ||
               ends_with_nocase(before, ".rar");
    }
    // name.r00, name.z01
    return ext.size() == 3 && (ext[0] == 'r' || ext[0] == 'z') &&
           ext.find_first_not_of("0123456789", 1) == std::string::npos;
}

bool is_cover_name(const std::string &name)
{
    const std::string low = lowered(name);
    return low.compare(0, 5, "icon0") == 0 && ends_with_nocase(low, ".png");
}

bool is_pkg(const std::string &name)
{
    return ends_with_nocase(name, ".pkg");
}

const NasGame *find_saved(const std::vector<NasGame> &saved, const std::string &entry)
{
    for (const NasGame &game : saved)
        if (game.entry == entry)
            return &game;
    return nullptr;
}

} // namespace

struct LibraryLoader::Job
{
    Connection connection;
    NasReader *nas = nullptr;   // the thread's way to the NAS (WebDAV or SMB)
    std::vector<NasGame> saved; // what the cache held
    bool key_art_only = false;  // a start_key_art() request
    NasGame target;             // ... for this game
    int ui_cpu = -1;
    std::atomic<bool> cancelled{false};
    std::atomic<int> owners{2}; // the loader and the thread; the last one deletes
    std::atomic<int> pictures{0}; // decoded pictures waiting for the UI
    pthread_mutex_t lock;
    std::deque<LibraryEvent> events;

    Job()
    {
        pthread_mutex_init(&lock, nullptr);
    }
    ~Job()
    {
        pthread_mutex_destroy(&lock);
    }

    void release()
    {
        if (owners.fetch_sub(1) == 1)
            delete this;
    }

    void post(LibraryEvent event)
    {
        // The app has one fixed heap: decoded covers (1 MB each) must not
        // pile up faster than the UI uploads them (two a frame), or a big
        // library runs it out of memory. So the reader waits here.
        const bool picture = !event.image.rgba.empty();
        if (picture)
        {
            while (pictures.load() >= kMaxQueuedPictures && !cancelled)
                usleep(4000);
            pictures.fetch_add(1);
        }
        pthread_mutex_lock(&lock);
        events.push_back(std::move(event));
        pthread_mutex_unlock(&lock);
    }

    void fail(const ProbeResult &result)
    {
        LibraryEvent event;
        event.kind = LibraryEvent::Kind::failed;
        event.headline = result.headline;
        event.detail = result.detail;
        post(std::move(event));
    }

    void progress(int done, int total, std::string text)
    {
        LibraryEvent event;
        event.kind = LibraryEvent::Kind::progress;
        event.done = done;
        event.total = total;
        event.detail = std::move(text);
        post(std::move(event));
    }

    void note(const std::string &entry, std::string text)
    {
        LibraryEvent event;
        event.kind = LibraryEvent::Kind::note;
        event.entry = entry;
        event.detail = std::move(text);
        post(std::move(event));
    }

    // A folder's entries (Depth 1); false when it can't be listed.
    bool list_folder(const std::string &url, std::vector<DavEntry> *entries,
                     NasProblem *problem = nullptr)
    {
        return nas->list(url, entries, problem);
    }

    // Reads parts of an image file on the NAS (for ExfatReader).
    ExfatReader::ReadAt image_reader(const std::string &path)
    {
        return [this, path](std::uint64_t offset, std::size_t length, std::string *out)
        {
            if (cancelled)
                return false;
            std::string why;
            return nas->read_range(path, offset, length, out, &why);
        };
    }

    // ---- folder games ----

    // A folder game's sce_sys, read from its listing: the stamp (param.json,
    // the cover and the key art as the NAS describes them), the names of the
    // cover and key art, and param.json's values.
    void read_sce_sys(const std::string &root, const std::vector<DavEntry> &inside, NasGame *game)
    {
        std::string stamp;
        std::string param_name;
        game->cover_file = {};
        game->art_file = {};
        for (const DavEntry &entry : inside)
        {
            if (entry.folder)
                continue;
            const std::string low = lowered(entry.name);
            const bool param = low == "param.json";
            const bool cover = is_cover_name(entry.name);
            const bool art = low == "pic0.png";
            if (!param && !cover && !art)
                continue;
            stamp += entry.name + ":" + stamp_of(entry) + ";";
            if (param)
                param_name = entry.name;
            if (cover && (game->cover_file.name.empty() || low == "icon0.png"))
                game->cover_file.name = entry.name;
            if (art)
                game->art_file.name = entry.name;
        }
        game->stamp = stamp;
        if (param_name.empty())
        {
            game->note = "the game's sce_sys folder has no param.json";
        }
        else
        {
            std::string json;
            NasProblem problem;
            if (nas->read_file(root + game->base + "/" + param_name, kMaxParam, &json, &problem))
                read_param_json(json, game);
            else
                game->note = "param.json: " + problem.brief;
        }
        if (game->cover_file.name.empty())
            game->cover_file.name = game->art_file.name; // the key art stands in
        if (game->cover_file.name.empty() && game->note.empty())
            game->note = "the game's sce_sys folder has no cover picture";
    }

    // Finds a game folder's sce_sys, breadth first, up to kDepth folders
    // down ("Game/PPSA01234-app/sce_sys"). A folder holding an .exfat image
    // instead becomes an image game. False when the folder holds no game at
    // all (a backport or patch folder, or an empty one): it isn't listed.
    bool read_folder(const std::string &root, NasGame *game)
    {
        struct Folder
        {
            std::string path; // under the games folder
            int depth;
        };
        std::vector<Folder> queue{{game->entry, 0}};
        DavEntry image;
        std::string image_path;
        // An sce_sys without pictures (a patch's, say) is kept only in case
        // no better one turns up deeper down.
        std::string fallback_base;
        std::vector<DavEntry> fallback_files;
        int listings = 0;
        for (std::size_t next = 0; next < queue.size() && listings < kListings; ++next)
        {
            if (cancelled)
                return false;
            const Folder folder = queue[next];
            std::vector<DavEntry> inside;
            NasProblem problem;
            ++listings;
            if (!list_folder(root + folder.path + "/", &inside, &problem))
            {
                if (folder.depth == 0)
                {
                    game->note = "the game's folder couldn't be listed: " + problem.brief;
                    return true; // can't tell: list it, with the reason
                }
                continue;
            }
            for (const DavEntry &entry : inside)
            {
                if (!entry.folder)
                {
                    if (image_path.empty() && ends_with_nocase(entry.name, ".exfat"))
                    {
                        image = entry;
                        image_path = folder.path + "/" + entry.name;
                    }
                    continue;
                }
                if (lowered(entry.name) == "sce_sys")
                {
                    const std::string base = folder.path + "/" + entry.name;
                    std::vector<DavEntry> sce_sys;
                    ++listings;
                    if (!list_folder(root + base + "/", &sce_sys))
                    {
                        if (fallback_base.empty())
                        {
                            fallback_base = base;
                            fallback_files.clear();
                        }
                        continue;
                    }
                    bool pictures = false;
                    for (const DavEntry &file : sce_sys)
                        pictures = pictures || (!file.folder && (is_cover_name(file.name) ||
                                                                 lowered(file.name) == "pic0.png"));
                    if (pictures)
                    {
                        game->base = base;
                        read_sce_sys(root, sce_sys, game);
                        return true;
                    }
                    if (fallback_base.empty())
                    {
                        fallback_base = base;
                        fallback_files = std::move(sce_sys);
                    }
                    continue;
                }
                if (folder.depth + 1 <= kDepth)
                    queue.push_back({folder.path + "/" + entry.name, folder.depth + 1});
            }
        }
        if (!fallback_base.empty())
        {
            game->base = fallback_base;
            if (fallback_files.empty())
                game->note = "the game's sce_sys folder couldn't be listed";
            else
                read_sce_sys(root, fallback_files, game);
            return true;
        }
        if (!image_path.empty())
        {
            // A folder around an image: it's an image game.
            game->image = true;
            game->image_path = image_path;
            game->size = image.size;
            game->size_known = true;
            game->stamp = stamp_of(image);
            return true;
        }
        return false;
    }

    // ---- image games ----

    // Looks inside an image: its sce_sys gives the name (param.json), the
    // cover and the key art. On failure game->note says why.
    void read_image(const std::string &root, NasGame *game)
    {
        ExfatReader reader(image_reader(root + game->image_path));
        std::string why;
        std::vector<ExfatFile> sce_sys;
        if (!reader.open(&why) || !reader.find_sce_sys(&sce_sys, &why))
        {
            if (!cancelled)
                game->note = "couldn't look inside the image: " + why;
            return;
        }
        game->inside = true;
        game->geometry = reader.geometry();
        if (const ExfatFile *param = find_entry(sce_sys, "param.json"))
        {
            std::string json;
            if (reader.read(*param, kMaxParam, &json, &why))
                read_param_json(json, game);
            else
                game->note = "param.json in the image: " + why;
        }
        const ExfatFile *cover = find_entry(sce_sys, "icon0.png");
        for (const ExfatFile &file : sce_sys)
            if (cover == nullptr && !file.directory && is_cover_name(file.name))
                cover = &file;
        const ExfatFile *art = find_entry(sce_sys, "pic0.png");
        if (cover == nullptr)
            cover = art;
        if (cover != nullptr)
            game->cover_file = *cover;
        if (art != nullptr)
            game->art_file = *art;
        if (cover == nullptr && game->note.empty())
            game->note = "the image's sce_sys has no cover picture";
    }

    // ---- pictures ----

    // One of a game's sce_sys pictures, from the NAS.
    bool fetch_picture(const std::string &root, const NasGame &game, const ExfatFile &file,
                       std::string *data, std::string *why)
    {
        if (file.name.empty())
            return false;
        if (game.image)
        {
            if (!game.inside)
                return false;
            ExfatReader reader(image_reader(root + game.image_path));
            reader.set_geometry(game.geometry);
            if (!reader.read(file, kMaxIcon, data, why))
            {
                *why = file.name + " in the image: " + *why;
                return false;
            }
            return true;
        }
        if (game.base.empty())
            return false;
        NasProblem problem;
        if (!nas->read_file(root + game.base + "/" + file.name, kMaxIcon, data, &problem))
        {
            *why = file.name + (problem.brief == "too big" ? " is too big" : ": " + problem.brief);
            return false;
        }
        return true;
    }

    // Fetches a game's cover and key art into the cache. The cover's bytes
    // are kept in *cover (to show it now); the key art only goes to storage.
    void fetch_pictures(const std::string &root, NasGame *game, std::string *cover)
    {
        std::string why;
        game->cover_saved = false;
        game->art_saved = false;
        if (fetch_picture(root, *game, game->cover_file, cover, &why))
            game->cover_saved = save_picture(cover_path(game->entry), *cover);
        else if (!why.empty() && game->note.empty())
            game->note = why;
        if (cancelled)
            return;
        std::string art;
        why.clear();
        if (game->art_file.name == game->cover_file.name && !cover->empty())
            art = *cover;
        else if (!fetch_picture(root, *game, game->art_file, &art, &why))
            return;
        game->art_saved = save_picture(art_path(game->entry), art);
    }

    // Reads the PKG directory table over SMB/WebDAV and extracts icon0.png
    // without downloading the package itself.
    bool fetch_pkg_cover(const std::string &root, const DavEntry &entry,
                         NasGame *game, std::string *cover)
    {
        // entry may live in any subfolder of the configured games root.
        // NasGame::entry is the path relative to that root, not just the basename.
        const std::string path = root + game->entry;
        auto be32 = [](const std::uint8_t *p) -> std::uint32_t {
            return (static_cast<std::uint32_t>(p[0]) << 24) |
                   (static_cast<std::uint32_t>(p[1]) << 16) |
                   (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
        };
        auto le64 = [](const std::uint8_t *p) -> std::uint64_t {
            std::uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (i * 8);
            return v;
        };
        auto read = [&](std::uint64_t offset, std::size_t length, std::string *out) {
            std::string why;
            return offset <= entry.size && length <= entry.size - offset &&
                   nas->read_range(path, offset, length, out, &why) && out->size() == length;
        };

        std::string header;
        if (!read(0, 0x200, &header)) {
            game->note = "PKG header could not be read";
            return false;
        }
        const auto *h = reinterpret_cast<const std::uint8_t *>(header.data());
        std::uint64_t cnt = 0;
        if (std::memcmp(h, "\x7f" "CNT", 4) == 0) {
            cnt = 0;
        } else if (std::memcmp(h, "\x7f" "LIH", 4) == 0) {
            cnt = le64(h + 0x30);
        } else if (std::memcmp(h, "\x7f" "FIH", 4) == 0) {
            cnt = le64(h + 0x58);
            std::string magic;
            if (cnt < 0x80 || !read(cnt, 4, &magic) || magic.compare(0, 4, "\x7f" "CNT", 4) != 0) {
                cnt = 0;
                for (std::size_t at = 0x10; at + 8 <= header.size(); at += 8) {
                    const std::uint64_t candidate = le64(h + at);
                    if (candidate < 0x10000 || candidate >= entry.size || (candidate & 0xfff))
                        continue;
                    if (read(candidate, 4, &magic) && magic.compare(0, 4, "\x7f" "CNT", 4) == 0) {
                        cnt = candidate;
                        break;
                    }
                }
            }
        }
        if (cnt >= entry.size || entry.size - cnt < 0x80) {
            game->note = "unsupported or invalid PKG header (CNT not found)";
            return false;
        }

        std::string ch;
        if (!read(cnt, 0x80, &ch) || ch.compare(0, 4, "\x7f" "CNT", 4) != 0) {
            game->note = "invalid PKG content header";
            return false;
        }
        const auto *c = reinterpret_cast<const std::uint8_t *>(ch.data());
        const std::uint32_t count = be32(c + 0x10);
        const std::uint32_t table_off = be32(c + 0x18);
        if (!count || count > 2048 || table_off > 0x200000 ||
            static_cast<std::uint64_t>(count) * 32 > entry.size - cnt ||
            table_off > entry.size - cnt - static_cast<std::uint64_t>(count) * 32) {
            game->note = "invalid PKG entry table";
            return false;
        }
        const std::size_t table_size = static_cast<std::size_t>(count) * 32;
        std::string table;
        if (!read(cnt + table_off, table_size, &table)) {
            game->note = "PKG entry table could not be read";
            return false;
        }
        const auto *t = reinterpret_cast<const std::uint8_t *>(table.data());
        std::uint32_t strings_off = 0, strings_size = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto *e = t + static_cast<std::size_t>(i) * 32;
            if (be32(e) == 0x0200) {
                strings_off = be32(e + 16);
                strings_size = be32(e + 20);
                break;
            }
        }
        std::string strings;
        if (strings_size && strings_size < 65536 &&
            read(cnt + strings_off, strings_size, &strings)) {
            // Names in the PKG string table are NUL-terminated.
        } else {
            strings.clear();
        }

        std::uint64_t icon_off = 0;
        std::uint32_t icon_size = 0;
        std::uint64_t param_off = 0;
        std::uint32_t param_size = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto *e = t + static_cast<std::size_t>(i) * 32;
            const std::uint32_t type = be32(e);
            const std::uint32_t name_off = be32(e + 4);
            const std::uint32_t data_off = be32(e + 16);
            const std::uint32_t data_size = be32(e + 20);
            std::string name;
            if (name_off < strings.size()) {
                const std::size_t end = strings.find('\0', name_off);
                if (end != std::string::npos) name = strings.substr(name_off, end - name_off);
            }
            if ((type == 0x2000 || name == "param.json") &&
                data_size > 0 && data_size < kMaxParam) {
                param_off = cnt + data_off;
                param_size = data_size;
            }
            if ((type == 0x1200 || name == "icon0.png") &&
                data_size > 8 && data_size <= kMaxIcon &&
                cnt + data_off <= entry.size && data_size <= entry.size - (cnt + data_off)) {
                icon_off = cnt + data_off;
                icon_size = data_size;
            }
        }
        if (param_size) {
            std::string json;
            if (read(param_off, param_size, &json))
                read_param_json(json, game);
        }
        if (!icon_size) {
            game->note = "PKG has no embedded icon0.png";
            return false;
        }
        if (!read(icon_off, icon_size, cover)) {
            game->note = "PKG icon0.png could not be read";
            return false;
        }
        game->note.clear();
        game->cover_saved = save_picture(cover_path(game->entry), *cover);
        return true;
    }

    // Decodes a cover and sends it to the UI.
    void show_cover(const std::string &entry, const std::string &data)
    {
        LibraryEvent event;
        event.kind = LibraryEvent::Kind::cover;
        event.entry = entry;
        std::string error;
        yield_now();
        if (!decode_png(reinterpret_cast<const std::uint8_t *>(data.data()), data.size(),
                        &event.image, &error))
        {
            note(entry, "the game's cover couldn't be read (" + error + ")");
            return;
        }
        if (event.image.width != event.image.height)
            crop_square(&event.image); // key art standing in: 1080 high
        shrink_to(&event.image, event.image.width > 1000 ? 600 : kCoverSide);
        cover_tones(event.image, event.tone);
        post(std::move(event));
        yield_now();
    }

    // ---- folder sizes ----

    // Adds up everything in a game folder. Archives (.zip, .rar, .7z and
    // their split parts) are left out: they are downloads waiting to be
    // unpacked, not part of the game. False when it can't tell.
    //
    // First choice: one "Depth: infinity" listing of the whole folder, read
    // as it streams in (deep_size). A big game's listing is tens of megabytes
    // of XML: holding it whole ran the app out of memory, so only the entry
    // being read is kept. When the NAS refuses deep listings, the folder is
    // listed one folder at a time instead, which is much slower. (SMB has no
    // deep listing: several folders are listed at once instead, over the one
    // sign-in.)
    bool measure(const std::string &url, std::uint64_t *total)
    {
        *total = 0;
        if (nas->smb())
        {
            int folders = 0;
            return nas->walk_smb(url,
                                 [&](const std::string &, const std::vector<DavEntry> &inside)
                                 {
                                     if (cancelled || ++folders > kSizeFolders)
                                         return false;
                                     for (const DavEntry &entry : inside)
                                         if (!entry.folder && !is_archive(entry.name))
                                             *total += entry.size;
                                     return true;
                                 });
        }
        if (deep_size(url, total))
            return true;
        *total = 0;
        std::vector<std::string> queue{url};
        for (std::size_t next = 0; next < queue.size(); ++next)
        {
            if (cancelled || static_cast<int>(queue.size()) > kSizeFolders)
                return false;
            std::vector<DavEntry> inside;
            if (!list_folder(queue[next], &inside))
                return false;
            for (const DavEntry &entry : inside)
            {
                if (entry.folder)
                    queue.push_back(queue[next] + entry.name + "/");
                else if (!is_archive(entry.name))
                    *total += entry.size;
            }
        }
        return true;
    }

    // One deep listing, summed as it arrives. False when the NAS won't give one.
    bool deep_size(const std::string &url, std::uint64_t *total)
    {
        *total = 0;
        std::string pending; // XML not yet read: at most part of one entry, plus a read
        int responses = 0;
        bool failed = false;
        const auto take = [&](bool final)
        {
            // Everything up to the last complete </...response> goes now.
            std::size_t cut = std::string::npos;
            for (std::size_t at = pending.rfind("response>"); at != std::string::npos;
                 at = at == 0 ? std::string::npos : pending.rfind("response>", at - 1))
            {
                const std::size_t open = pending.rfind('<', at);
                if (open != std::string::npos && open + 1 < pending.size() && pending[open + 1] == '/')
                {
                    cut = at + 9;
                    break;
                }
            }
            if (cut == std::string::npos)
            {
                if (final)
                    pending.clear();
                return;
            }
            std::vector<DavEntry> entries;
            if (parse_listing(pending.substr(0, cut), url, &entries) || cut > 0)
                ++responses;
            for (const DavEntry &entry : entries)
                if (!entry.folder && !is_archive(entry.name))
                    *total += entry.size;
            pending.erase(0, cut);
        };
        const HttpReply reply = http_stream(
            connection, "PROPFIND", url,
            "Depth: infinity\r\nContent-Type: application/xml; charset=utf-8\r\n", propfind_body(),
            [&](const char *data, std::size_t length)
            {
                if (cancelled)
                    return false;
                pending.append(data, length);
                if (pending.size() >= 256 * 1024)
                    take(false);
                if (pending.size() > 8u * 1024u * 1024u)
                {
                    failed = true; // one entry this long isn't a listing
                    return false;
                }
                return true;
            });
        if (cancelled || failed || reply.status != 207 || !reply.complete)
            return false; // refused, or cut short: a partial sum would be wrong
        take(true);
        return responses > 0;
    }

    // ---- the whole library ----

    void run()
    {
        const std::string root = folder_path(connection);

        // 1. The saved covers, from the PS5's storage: the library looks
        //    complete before the NAS has answered.
        int step = 0;
        std::map<std::string, bool> lost; // saved covers that weren't in storage after all
        for (const NasGame &game : saved)
        {
            if (cancelled)
                return;
            std::string data;
            if (game.cover_saved && load_picture(cover_path(game.entry), &data, kMaxIcon))
                show_cover(game.entry, data);
            else if (game.cover_saved)
                lost[game.entry] = true;
            progress(++step, static_cast<int>(saved.size()), "Loading your library");
        }

        // 2. What's on the NAS now.
        progress(0, 0, "Checking your NAS");
        NasProblem listing;
        std::vector<DavEntry> entries;
        const bool listed = list_folder(root, &entries, &listing);
        if (cancelled)
            return;
        if (!listed)
        {
            fail(explain_nas_failure(connection, listing));
            return;
        }
        std::vector<DavEntry> candidates;
        // Keep game folders, images and packages at the configured root as
        // before. In addition, walk every visible subfolder and collect PKGs
        // at any depth (e.g. PS4/Action/Game/update.pkg). Paths are kept
        // relative to root so cache keys remain unique for equal filenames.
        for (const DavEntry &entry : entries)
            if (entry.folder || ends_with_nocase(entry.name, ".exfat") || is_pkg(entry.name))
                candidates.push_back(entry);

        // SMB has a batched, multi-sign-in tree walker. Use it instead of
        // issuing one synchronous directory request at a time: on a NAS with
        // many subfolders the latter can look like a hung library scan.
        // WebDAV has no equivalent, so retain the bounded breadth-first walk.
        // Use the same sequential, bounded traversal for SMB and WebDAV.
        // The parallel SMB walker queued every discovered directory at once;
        // on large Synology trees this could grow memory use until the PS5
        // terminated the app. One directory per request keeps memory bounded.
        struct PendingFolder
        {
            std::string path;
            int depth;
        };
        std::deque<PendingFolder> folders;
        for (const DavEntry &entry : entries)
            if (entry.folder && !entry.name.empty() && entry.name[0] != '.' &&
                lowered(entry.name) != "@eadir" && lowered(entry.name) != "#recycle" &&
                lowered(entry.name) != "lost+found")
                folders.push_back({entry.path, 1});

        // Keep the scan responsive on large SMB shares. A bounded walk still
        // finds packages in ordinary nested layouts without flooding the PS5
        // with thousands of synchronous directory requests.
        constexpr int kPkgWalkDepth = 8;
        constexpr int kPkgWalkFolders = 250;
        constexpr std::size_t kPkgWalkPending = 512;
        int walked = 0;
        bool scan_limited = false;
        while (!folders.empty() && walked < kPkgWalkFolders && !cancelled)
        {
            const PendingFolder folder = std::move(folders.front());
            folders.pop_front();
            if (folder.depth > kPkgWalkDepth)
                continue;

            std::vector<DavEntry> inside;
            if (!list_folder(folder.path + "/", &inside))
                continue;
            ++walked;

            for (const DavEntry &entry : inside)
            {
                if (entry.folder)
                {
                    if (folder.depth < kPkgWalkDepth && !entry.name.empty() &&
                        entry.name[0] != '.' && lowered(entry.name) != "@eadir" &&
                        lowered(entry.name) != "#recycle" && lowered(entry.name) != "lost+found")
                    {
                        if (folders.size() < kPkgWalkPending)
                            folders.push_back({entry.path, folder.depth + 1});
                        else
                            scan_limited = true;
                    }
                    continue;
                }
                if (!is_pkg(entry.name))
                    continue;
                if (entry.path.compare(0, root.size(), root) != 0)
                    continue;
                const std::string relative = entry.path.substr(root.size());
                if (relative.empty())
                    continue;
                const bool already = std::any_of(
                    candidates.begin(), candidates.end(),
                    [&](const DavEntry &candidate)
                    {
                        return !candidate.folder && candidate.path == entry.path;
                    });
                if (!already)
                    candidates.push_back(entry);
            }
            if ((walked % 10) == 0)
                progress(walked, kPkgWalkFolders,
                         "Scanning " + folder.path.substr(root.size()));
        }
        if (!folders.empty() || walked >= kPkgWalkFolders)
            scan_limited = true;
        if (scan_limited)
            progress(walked, kPkgWalkFolders,
                     "Scan limit reached; showing discovered PKG files");
        if (cancelled)
            return;

        // 3. Each game: unchanged ones come from the cache, the rest are read.
        std::vector<NasGame> games;
        std::map<std::string, std::string> fresh_covers; // entry -> cover bytes
        const int total = static_cast<int>(candidates.size());
        for (int k = 0; k < total; ++k)
        {
            if (cancelled)
                return;
            const DavEntry &entry = candidates[static_cast<std::size_t>(k)];
            const std::string relative_entry =
                entry.path.compare(0, root.size(), root) == 0
                    ? entry.path.substr(root.size())
                    : entry.name;
            progress(k, total, "Checking " + relative_entry);
            const NasGame *before = find_saved(saved, relative_entry);
            const std::string folder_stamp = entry.folder ? stamp_of(entry) : "";
            NasGame game;
            bool same = false;
            if (before != nullptr)
            {
                if (!entry.folder)
                {
                    same = before->stamp == stamp_of(entry) &&
                           (is_pkg(entry.name) ? !before->image : before->image);
                }
                else if (before->image)
                {
                    same = before->folder_stamp == folder_stamp; // an image in a folder
                }
                else if (!before->base.empty())
                {
                    std::vector<DavEntry> sce_sys;
                    if (list_folder(root + before->base + "/", &sce_sys))
                    {
                        std::string stamp;
                        for (const DavEntry &file : sce_sys)
                        {
                            const std::string low = lowered(file.name);
                            if (!file.folder &&
                                (low == "param.json" || low == "pic0.png" || is_cover_name(file.name)))
                                stamp += file.name + ":" + stamp_of(file) + ";";
                        }
                        same = stamp == before->stamp;
                    }
                }
            }
            if (same)
            {
                game = *before;
            }
            else
            {
                game.entry = relative_entry;
                game.title = entry.folder ? entry.name : without_extension(entry.name);
                if (!entry.folder)
                {
                    game.size = entry.size;
                    game.size_known = true;
                    game.stamp = stamp_of(entry);
                    if (is_pkg(entry.name))
                    {
                        game.image = false;
                        game.note = "PKG package listed; checking embedded cover.";
                    }
                    else
                    {
                        game.image = true;
                        game.image_path = entry.name;
                    }
                }
                else if (!read_folder(root, &game))
                {
                    continue; // not a game (a backport, a patch, an empty folder)
                }
                if (game.image)
                    read_image(root, &game);
            }
            if (cancelled)
                return;
            if (entry.folder && !game.image)
            {
                // The folder's size is measured again when it changed.
                if (!(same && game.folder_stamp == folder_stamp && game.size_known))
                {
                    game.size = 0;
                    game.size_known = false;
                }
                game.folder_stamp = folder_stamp;
            }
            else if (entry.folder)
            {
                game.folder_stamp = folder_stamp;
            }
            // Pictures: fetched when the game is new or changed, or when the
            // cache lost them.
            if (is_pkg(entry.name))
            {
                const bool cover_missing = !game.cover_saved || !same || lost.count(game.entry) > 0;
                if (cover_missing) {
                    std::string cover;
                    game.cover_saved = false;
                    if (fetch_pkg_cover(root, entry, &game, &cover) && !cover.empty())
                        fresh_covers[game.entry] = std::move(cover);
                }
            }
            else
            {
                const bool cover_missing = !game.cover_file.name.empty() &&
                                           (!game.cover_saved || !same || lost.count(game.entry) > 0);
                const bool art_missing = !game.art_file.name.empty() && (!game.art_saved || !same);
                if (cover_missing || art_missing)
                {
                    std::string cover;
                    fetch_pictures(root, &game, &cover);
                    if (!cover.empty())
                        fresh_covers[game.entry] = std::move(cover);
                }
            }
            games.push_back(std::move(game));
        }
        std::sort(games.begin(), games.end(), title_before);
        for (const NasGame &old : saved)
            if (find_saved(games, old.entry) == nullptr)
                forget_pictures(old.entry);
        {
            LibraryEvent event;
            event.kind = LibraryEvent::Kind::listed;
            event.games = games;
            post(std::move(event));
        }
        for (const auto &[entry, data] : fresh_covers)
        {
            if (cancelled)
                return;
            show_cover(entry, data);
        }
        fresh_covers.clear();
        save_library(connection, games);
        {
            LibraryEvent event;
            event.kind = LibraryEvent::Kind::ready;
            post(std::move(event));
        }

        // 4. Folder sizes, for the folders that changed or were never measured.
        std::vector<std::size_t> to_measure;
        for (std::size_t i = 0; i < games.size(); ++i)
            if (!games[i].image && !games[i].size_known)
                to_measure.push_back(i);
        for (std::size_t k = 0; k < to_measure.size(); ++k)
        {
            if (cancelled)
                return;
            NasGame &game = games[to_measure[k]];
            progress(static_cast<int>(k), static_cast<int>(to_measure.size()),
                     "Measuring " + game.title);
            std::uint64_t bytes = 0;
            if (!measure(root + game.entry + "/", &bytes))
                continue;
            game.size = bytes;
            game.size_known = true;
            LibraryEvent event;
            event.kind = LibraryEvent::Kind::size;
            event.entry = game.entry;
            event.bytes = bytes;
            post(std::move(event));
        }
        if (!to_measure.empty() && !cancelled)
            save_library(connection, games);
    }

    // ---- one game's key art ----

    void run_key_art()
    {
        const std::string root = folder_path(connection);
        std::string data;
        if (!load_picture(art_path(target.entry), &data, kMaxIcon))
        {
            std::string why;
            if (!fetch_picture(root, target, target.art_file, &data, &why))
                return;
            save_picture(art_path(target.entry), data);
        }
        if (cancelled)
            return;
        LibraryEvent event;
        event.kind = LibraryEvent::Kind::key_art;
        event.entry = target.entry;
        std::string error;
        yield_now();
        if (!decode_png(reinterpret_cast<const std::uint8_t *>(data.data()), data.size(),
                        &event.image, &error))
            return;
        shrink_to(&event.image, 1920);
        post(std::move(event));
    }

    // Starts the job's thread; on failure posts why, and the job has one owner.
    static void launch(Job *job)
    {
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setstacksize(&attributes, 1024 * 1024);
        pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
        pthread_t thread{};
        const int created = pthread_create(&thread, &attributes, &Job::thread_main, job);
        pthread_attr_destroy(&attributes);
        if (created != 0)
        {
            LibraryEvent event;
            event.kind = LibraryEvent::Kind::failed;
            event.headline = "Couldn't start reading the NAS";
            event.detail = "Close NASBridge and open NASBridge again.";
            job->post(std::move(event));
            LibraryEvent done;
            done.kind = LibraryEvent::Kind::finished;
            job->post(std::move(done));
            job->owners.store(1); // no thread will release it
        }
    }

    static void *thread_main(void *self)
    {
        Job *job = static_cast<Job *>(self);
        keep_off_cpu(job->ui_cpu);
        {
            NasReader nas(job->connection);
            job->nas = &nas;
            if (job->key_art_only)
                job->run_key_art();
            else
                job->run();
            job->nas = nullptr;
        }
        LibraryEvent done;
        done.kind = LibraryEvent::Kind::finished;
        job->post(std::move(done));
        job->release();
        return nullptr;
    }
};

LibraryLoader::~LibraryLoader()
{
    stop();
}

void LibraryLoader::start(const Connection &connection, const std::vector<NasGame> &saved)
{
    stop();
    Job *job = new Job();
    job->connection = connection;
    job->saved = saved;
    job->ui_cpu = current_cpu();
    job_ = job;
    Job::launch(job);
}

void LibraryLoader::start_key_art(const Connection &connection, const NasGame &game)
{
    stop();
    Job *job = new Job();
    job->connection = connection;
    job->key_art_only = true;
    job->target = game;
    job->ui_cpu = current_cpu();
    job_ = job;
    Job::launch(job);
}

void LibraryLoader::stop()
{
    if (job_ == nullptr)
        return;
    job_->cancelled = true;
    job_->release();
    job_ = nullptr;
}

bool LibraryLoader::take(LibraryEvent *event)
{
    if (job_ == nullptr)
        return false;
    pthread_mutex_lock(&job_->lock);
    const bool any = !job_->events.empty();
    if (any)
    {
        *event = std::move(job_->events.front());
        job_->events.pop_front();
        if (!event->image.rgba.empty())
            job_->pictures.fetch_sub(1);
    }
    pthread_mutex_unlock(&job_->lock);
    if (any && event->kind == LibraryEvent::Kind::finished)
        stop(); // the thread has let go; this frees the job
    return any;
}

} // namespace hui::concepts

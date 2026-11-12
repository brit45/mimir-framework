local FS = dofile(ROOTWORK .. "/scripts/modules/fs.lua")
local root = os.tmpname()
os.remove(root)
assert(FS.mkdir_p(FS.join(root, "data")))
assert(FS.mkdir_p(FS.join(root, "empty")))
local data_dir = FS.join(root, "data")
for sample = 1, 12 do
  local stem = FS.join(data_dir, string.format("sample_%02d", sample))
  local image = assert(io.open(stem .. ".ppm", "wb"))
  image:write("P6\n4 2\n255\n")
  for pixel = 1, 24 do image:write(string.char((sample * 13 + pixel) % 256)) end
  image:close()
  local text = assert(io.open(stem .. ".txt", "wb"))
  text:write("Image ", sample, ": texte UTF-8, été."); text:close()
end
local function load()
  local ok, count = Mimir.Dataset.load(data_dir, 4, 2, 2, true, FS.join(root, "cache.json"), 32, true)
  assert(ok, count); assert(count == 12, count)
end
load()
local expected = {}
for index = 1, 12 do
  local item = assert(Mimir.Dataset.get(index, false))
  assert(item.image == nil and item.mask == nil)
  local source_index = assert(tonumber(item.image_file:match("sample_(%d+)%.ppm$")))
  expected[index] = {source_index = source_index, text = item.text, image_file = item.image_file}
  assert(Mimir.Dataset.release(index))
end
-- JSON checkpoint readers restore integral values as Lua floats.
local order = {7.0, 2.0, 2.0, 11.0, 1.0}
assert(math.type(order[1]) == "float")
local ok, count = Mimir.Dataset.prefetch(order)
assert(ok, count); assert(count == #order)
if Mimir.Dataset.prefetch_stats().enabled then
  local deadline = os.clock() + 5
  while Mimir.Dataset.prefetch_stats().ready < 1 do assert(os.clock() < deadline, "producer stalled") end
end
for _, index in ipairs(order) do
  local item = assert(Mimir.Dataset.get(index))
  assert(item.text == expected[index].text and item.image_file == expected[index].image_file)
  assert(item.width == 4 and item.height == 2 and item.channels == 3)
  assert(#item.image == 24 and item.frame_count == 1 and not item.is_animated)
  for pixel = 1, 24 do assert(item.image[pixel] == (expected[index].source_index * 13 + pixel) % 256) end
  assert(Mimir.Dataset.release(index))
end
local stats = Mimir.Dataset.prefetch_stats()
if stats.enabled then
  assert(stats.hits >= 1 and stats.prepared >= #order - 1)
  assert(stats.queued + stats.ready + stats.leased + stats.loading <= stats.capacity + 1)
end
assert(not Mimir.Dataset.prefetch({0}))
assert(not Mimir.Dataset.prefetch({1.5}))
assert(not Mimir.Dataset.prefetch({"1"}))
assert(not Mimir.Dataset.prefetch({math.huge}))
assert(not Mimir.Dataset.prefetch("bad"))
assert(not Mimir.Dataset.release(13))
assert(Mimir.Dataset.prefetch({}))
-- Stop/reload while the producer may still be preparing old requests.
load()
assert(Mimir.Dataset.get(1).image)
local builder = Mimir.Database.load()
local loaded, n = builder.cache(data_dir, 4, 2, 2, FS.join(root, "builder-cache.json"), 32, true)
assert(loaded, n); assert(n == 12)
assert(Mimir.Dataset.prefetch({12, 1}))
assert(Mimir.Dataset.get(12).image)
assert(Mimir.Dataset.load(FS.join(root, "empty"), 4, 2))
assert(Mimir.Dataset.get(1) == nil)
assert(os.execute((FS.is_windows() and "rmdir /s /q " or "rm -rf ") .. FS.quote(root)))
print("DATASET_PREFETCH_LUA_OK ordered duplicates, RGB/text parity, metadata-only, reload/cache, invalid indices, shutdown")

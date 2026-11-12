-- Neovim 0.11+: load with :luafile /path/to/mimir/ide/neovim/mimir.lua
-- Install lua-language-server, typescript-language-server and rust-analyzer separately.
local file = debug.getinfo(1, 'S').source:sub(2)
local root = vim.fs.dirname(vim.fs.dirname(vim.fs.dirname(vim.fs.normalize(file))))
vim.lsp.config('lua_ls', {
  cmd = { 'lua-language-server' },
  filetypes = { 'lua' },
  root_markers = { '.luarc.json', '.git' },
  settings = { Lua = { workspace = { library = { root .. '/ide/lua' }, checkThirdParty = false },
    runtime = { version = 'Lua 5.3' } } },
})
vim.lsp.config('ts_ls', {
  cmd = { 'typescript-language-server', '--stdio' },
  filetypes = { 'javascript', 'javascriptreact', 'typescript' },
  root_markers = { 'jsconfig.json', 'tsconfig.json', '.git' },
})
vim.lsp.config('rust_analyzer', {
  cmd = { 'rust-analyzer' }, filetypes = { 'rust' }, root_markers = { 'Cargo.toml', '.git' },
  settings = { ['rust-analyzer'] = { linkedProjects = { root .. '/ide/rust/Cargo.toml' } } },
})
vim.lsp.enable({ 'lua_ls', 'ts_ls', 'rust_analyzer' })

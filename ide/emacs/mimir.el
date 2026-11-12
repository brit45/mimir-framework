;;; mimir.el --- Mimir project Eglot helpers -*- lexical-binding: t; -*-
;; Load this file; language servers must already be installed.
(require 'eglot)
(defconst mimir-ide-root
  (expand-file-name "../.." (file-name-directory (or load-file-name buffer-file-name))))
(defun mimir-ide-setup ()
  "Configure Eglot only for files in this Mimir checkout."
  (when (and buffer-file-name (file-in-directory-p buffer-file-name mimir-ide-root))
    (setq-local eglot-workspace-configuration
                `(:Lua (:runtime (:version "Lua 5.3")
                        :workspace (:library [,(expand-file-name "ide/lua" mimir-ide-root)]
                                    :checkThirdParty :json-false))
                  :rust-analyzer (:linkedProjects [,(expand-file-name "ide/rust/Cargo.toml" mimir-ide-root)])))))
(add-hook 'prog-mode-hook #'mimir-ide-setup)
(add-to-list 'eglot-server-programs '(lua-mode . ("lua-language-server")))
(add-to-list 'eglot-server-programs '(rust-mode . ("rust-analyzer")))
(add-to-list 'eglot-server-programs '(js-mode . ("typescript-language-server" "--stdio")))
(provide 'mimir)

(require 'package)
(setq package-user-dir (expand-file-name "./.packages"))
(setq package-archives '(("melpa" . "https://melpa.org/packages/")
                         ("elpa" . "https://elpa.gnu.org/packages/")))

;; Initialize the package system
(package-initialize)
(unless package-archive-contents
  (package-refresh-contents))


(package-install 'htmlize)
(package-install 'org)

(require 'ox-publish)

(setq org-html-validation-link nil ;; Do not show "Validate" link
      org-confirm-babel-evaluate nil)

(setq org-publish-project-alist
      '(("Pollsposition"
         :base-directory "org"
         :base-extension "org"
         :publishing-directory "_public/"
         :publishing-function org-html-publish-to-html
         :with-broken-links f
         :recursive f
         :author "Rémi Louf, Alexandre Andorra"
         :email "bonjour@pollsposition.com"
         :with-toc nil
         :html-postamble nil
         :section-numbers nil
         :htmlized-source t
         :html-head-extra "<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" /><script data-goatcounter=\"https://thetypicalset.goatcounter.com/count\" async src=\"//gc.zgo.at/count.js\"></script>"
         :auto-sitemap t
         :sitemap-filename "index.html")))

(defun my/publish-all()
  (call-interactively 'org-publish-all))

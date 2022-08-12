publish: publish.el
	@echo "Exporting the project to HTML with org-publish"
	emacs --batch -q --load publish.el --funcall my/publish-all

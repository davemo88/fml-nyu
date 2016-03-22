(TeX-add-style-hook
 "hw1"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("babel" "english")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "fontenc"
    "geometry"
    "float"
    "babel"
    "amsfonts"
    "amsmath"
    "bm"
    "tikz"
    "algorithm"
    "algorithmic")
   (LaTeX-add-labels
    "fig:C3"))
 :latex)


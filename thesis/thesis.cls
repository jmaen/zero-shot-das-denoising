\ProvidesClass{thesis}[]

\LoadClass[a4paper,11pt,twoside,english]{book}
\usepackage[a4paper,left=3cm,right=3cm,bottom=3cm,top=3cm]{geometry}
\renewcommand{\baselinestretch}{1.25}
\pagestyle{plain}
\usepackage{emptypage}
\setlength\parindent{0pt}

\usepackage[utf8]{inputenc}
\usepackage[english]{babel}

% Useful packages
\usepackage{graphicx}
\usepackage{color}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}

% Bibliography
\usepackage{biblatex}
\usepackage{csquotes}
\addbibresource{references.bib}

% Math shorthands
\newcommand{\normal}{\mathcal{N}(0,\sigma^2\mathbf{I})}

\newcommand{\R}{\mathbb{R}}
% \newcommand{\C}{\mathbb{C}}

% \newcommand{\E}[2]{\mathbb{E}_{#1}\left[#2\right]}
% \newcommand{\cnormal}[1][z]{\mathcal{N}(#1\mid 0,\mathbf{I})}

% \DeclareMathOperator*{\argmax}{argmax}
% \DeclareMathOperator*{\argmin}{argmin}

% Avoid widows and orphans
\clubpenalty = 10000
\widowpenalty = 10000 
\displaywidowpenalty = 10000

\ProvidesClass{thesis}[]

% Layout
\LoadClass[a4paper,11pt,twoside,english]{book}
\usepackage[a4paper,left=3cm,right=3cm,bottom=3cm,top=3cm]{geometry}
\renewcommand{\baselinestretch}{1.25}
\pagestyle{plain}
\usepackage{emptypage}
\setlength\parindent{0pt}
\setlength{\columnsep}{20pt}

% Base packages
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage[dvipsnames]{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{amsmath,amssymb}

% Figures
\usepackage{wrapfig}
\usepackage{graphicx}
\usepackage[labelfont=bf]{caption}
\usepackage{tikz}
\usepackage{tikz-3dplot}
\usetikzlibrary{decorations.pathreplacing}
% \usepackage{pgfplots}

% Bibliography
\usepackage{csquotes}
\usepackage{biblatex}
\addbibresource{references.bib}

% Abbreviations
\newcommand{\etal}{et al.\ }
\newcommand{\eg}{e.g.\ }

% Math shorthands
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}

\newcommand{\normal}{\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})}

% \newcommand{\E}[2]{\mathbb{E}_{#1}\left[#2\right]}
% \newcommand{\cnormal}[1][z]{\mathcal{N}(#1\mid 0,\mathbf{I})}

% \DeclareMathOperator*{\argmax}{argmax}
% \DeclareMathOperator*{\argmin}{argmin}

% Avoid widows and orphans
\clubpenalty = 10000
\widowpenalty = 10000 
\displaywidowpenalty = 10000

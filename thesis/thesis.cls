\ProvidesClass{thesis}[]

\LoadClass[a4paper,11pt,twoside,english]{book}
\pagestyle{plain}
\usepackage{emptypage}

\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage[a4paper,left=3cm,right=3cm,bottom=3cm,top=3cm]{geometry}

% Useful packages
\usepackage{graphicx}
\usepackage{color}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}

% Bibliography
\usepackage{biblatex}
\addbibresource{references.bib}

% Math shorthands
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}

\newcommand{\E}[2]{\mathbb{E}_{#1}\left[#2\right]}
\newcommand{\normal}{\mathcal{N}(0,\mathbf{I})}
\newcommand{\cnormal}[1][z]{\mathcal{N}(#1\mid 0,\mathbf{I})}

\DeclareMathOperator*{\argmax}{argmax}
\DeclareMathOperator*{\argmin}{argmin}

% Keine einzelnen Zeilen beim Anfang eines Abschnitts (Schusterjungen)
\clubpenalty = 10000
% Keine einzelnen Zeilen am Ende eines Abschnitts (Hurenkinder)
\widowpenalty = 10000 \displaywidowpenalty = 10000

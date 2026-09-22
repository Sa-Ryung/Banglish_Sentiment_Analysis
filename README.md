
<a id="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">Banglish Sentiment Analysis</h3>

  <p align="center">
    A comparative study of machine learning approaches for sentiment classification of Banglish (Bengali-English code-mixed) text using Bag-of-Words and TF-IDF representations.
    <br />
    <a href="https://github.com/Sa-Ryung/Banglish_Sentiment_Analysis"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Sa-Ryung/Banglish_Sentiment_Analysis/issues">Report Bug</a>
    ·
    <a href="https://github.com/Sa-Ryung/Banglish_Sentiment_Analysis/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#methodology">Methodology & Pipeline</a></li>
    <li><a href="#results">Results & Key Findings</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Code-mixing—the blending of two or more languages within conversation or text—is a widespread phenomenon in digital communication across South Asia. In Bangladesh, informal online discussions heavily feature **"Banglish"**, where Bengali words are transliterated into the Latin alphabet and freely mixed with English terms, abbreviations, and emojis. 

This repository houses an R-based framework implementing a custom preprocessing pipeline and comparing classical feature engineering techniques (**Bag-of-Words** and **TF-IDF**) alongside generative and discriminative classifiers (**Naïve Bayes** and regularized **Multinomial Logistic Regression**).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![R][R-shield]][R-url]
* **tm** (Text Mining Package)
* **caret** (Classification And REgression Training)
* **glmnet** (Regularized Generalized Linear Models)[cite: 1]
* **e1071** (Misc Functions of the Department of Statistics, Probability Theory Group)[cite: 1]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

You need R installed on your system along with RStudio. The script automatically checks for and installs any missing required packages (`tm`, `dplyr`, `caTools`, `stringr`, `e1071`, `caret`, `glmnet`, `vioplot`, `fmsb`, `Matrix`, `naivebayes`)[cite: 1].

### Installation

1. Clone the repo
   ```bash
   git clone [https://github.com/Sa-Ryung/Banglish_Sentiment_Analysis.git](https://github.com/Sa-Ryung/Banglish_Sentiment_Analysis.git)

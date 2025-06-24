# Visualizing Design of Experiments with ViDoe: An Interactive Approach

Data is a central asset in decision-making processes, especially when reliability and efficiency are key. Understanding how to design experiments and analyze their results is a practical skill that helps students, researchers, and professionals translate questions into structured insights. This e-book presents the foundations of a web application called **ViDoe**—*Visualizing Design of Experiments*—developed by Professors Leonardo H. Talero-Sarmiento, Henry Lamos-Díaz, and engineer David Márquez-González.

**ViDoe** is a user-friendly web platform created to support the learning and teaching of design of experiments (DoE). Through interactive tools, it allows users to generate datasets, apply experimental designs, and visualize results dynamically. It is intended for those who want to explore core ideas behind statistical modeling and hypothesis testing, without needing advanced knowledge in mathematics or programming.

This e-book does not aim to cover the entire theory or taxonomy of DoE, nor does it provide a detailed compendium of theorems. Instead, it introduces the main principles and structures that support practical experimentation. Its focus is on using **ViDoe** as an applied tool to foster understanding and experimentation. While the theoretical depth required for advanced study is beyond the scope of this material, readers looking for a more rigorous foundation are encouraged to consult **Montgomery’s “Design and Analysis of Experiments”**, a standard reference in the field.

We recognize that this quantitative approach is not commonly emphasized in many undergraduate programs. In fact, it is often sought after by graduate students working on process improvement or experimental design in engineering and related fields. With this in mind, the handbook offers a bridge between basic concepts and interactive practice, helping users gain confidence in structuring and analyzing experiments.

Whether you are just beginning to explore the world of experimental design or looking for a practical companion to complement your theoretical learning, **ViDoe** and this handbook are designed to support your journey in a clear, structured, and hands-on way.

---

## Table of Contents

**Section 1. Introduction to ViDoe**

- [Overview of ViDoe](#00-ViDoe-Introduction)  
- [One-Way ANOVA using ViDoe](#00-ViDoe-ANOVA-One-Way)  
- [Factorial Designs in ViDoe](#00-ViDoe-Factorial-Designs)

**Section 2. One-Way ANOVA**

- [Foundations of One-Way ANOVA](#01-anova-oneway)  
- [Notebook: 01-one-way-anova.ipynb](#01-one-way-anova)  
- [Notebook: 01-one-way-example.ipynb](#01-one-way-example)

**Section 3. Factorial Experimental Designs**

- [Principles of \(2^k\) and Fractional Factorial Designs](#00-ViDoe-Factorial-Designs)

**Section 4. Taguchi Designs**

- [Taguchi Methods and Robust Design](#05-Taguchi-Designs)

**Section 5. Block Designs**

- [Explanation of Block Designs](#04-block-designs)

**Section 6. Output Analysis**

- [Boxplot by Group](#03-Boxplot-by-group)  
- [Surface Plot](#03-Surface-plot)  
- [Regression Model](#03-regression-model)  
- [Residual Analysis](#03-residual-analysis)

**Extras**

- [Bibliography](#Bibliography)

---

## Short Biographies

**Leonardo H. Talero-Sarmiento** holds a Ph.D. in Engineering from Universidad Autónoma de Bucaramanga. His academic focus includes mathematical modeling, data analytics, operations research, process improvement, and technology adoption. He has authored scientific articles in journals such as *Digital Policy, Regulation and Governance*, *Heliyon*, *Revista Colombiana de Computación*, *Suma de Negocios*, *IngeCUC*, *Apuntes del Cenes*, *Estudios Gerenciales*, and *Contaduría y Administración*.  
📧 [ltalero@unab.edu.co](mailto:ltalero@unab.edu.co)  
🔗 [ResearchGate Profile](https://www.researchgate.net/profile/Leonardo-Talero?ev=hdr_xprf)  
🔗 [Institutional Profile](https://apolo.unab.edu.co/en/persons/leonardo-hernan-talero-sarmiento-3)

**Henry Lamos-Díaz** is a Full Professor with a Ph.D. in Mathematical Physics from the State University of Moscow. He also holds Master’s degrees in Computer Science from Universidad Industrial de Santander and in Mathematics from Patricio Lumumba University, where he completed his undergraduate studies. His research focuses on modeling, simulation, and optimization of production and logistics systems, with particular interest in humanitarian logistics.  
📧 [hlamos@uis.edu.co](mailto:hlamos@uis.edu.co)

**David Márquez-González** holds a Master's degree in Industrial Engineering from Universidad Industrial de Santander. His work focuses on modeling and optimizing production and logistics systems, with a strong emphasis on improving agricultural systems. He also works as a consultant in production optimization.  
📧 [juan2208424@correo.uis.edu.co](mailto:juan2208424@correo.uis.edu.co)

---

```{figure} images/INGENIERÍA-INDUSTRIAL_LOGO.png
---
alt: Industrial Engineering Program – Universidad Autónoma de Bucaramanga
---
```

## Note:
The ViDoe platform and its conceptual framework were originally presented at the **INNODOCT 2023 International Conference on Innovation, Documentation, Education and Teaching Technologies**, held in Bucaramanga. The paper details the motivation, technical approach, and educational goals behind the development of the tool. Readers may cite this work as follows:

> Talero-Sarmiento, L. H., Lamos-Díaz, H., & Márquez-González, J. D. (2024). *ViDoe: A novel tool for visualizing Design of Experiments*. In **INNODOCT 2023: Proceedings of the International Conference on Innovation, Documentation, Education and Teaching Technologies** (Vol. 11, pp. 88–95). Bucaramanga, Colombia.  
> [https://www.researchgate.net/publication/378149141_ViDoe_A_novel_tool_for_visualizing_Design_of_Experiments](https://www.researchgate.net/publication/378149141_ViDoe_A_novel_tool_for_visualizing_Design_of_Experiments)

### How to cite this work
```bibtex
@inproceedings{Talero2024ViDoe,
  author       = {Leonardo H. Talero-Sarmiento and Henry Lamos-Díaz and Juan David Márquez-González},
  title        = {{ViDoe: A novel tool for visualizing Design of Experiments}},
  booktitle    = {Proceedings of the International Conference on Innovation, Documentation, Education and Teaching Technologies (INNODOCT 2023)},
  year         = {2024},
  volume       = {11},
  pages        = {88--95},
  address      = {Bucaramanga, Colombia},
  url          = {https://www.researchgate.net/publication/378149141_ViDoe_A_novel_tool_for_visualizing_Design_of_Experiments}
}

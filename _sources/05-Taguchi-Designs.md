<a name="05-Taguchi-Designs"></a>
# Taguchi Designs

```{contents}
```

## 1. Introduction to Taguchi Methods

Taguchi Methods, developed by Dr. Genichi Taguchi, are statistical methods, or more broadly, a quality engineering philosophy, aimed at improving the quality of manufactured goods and, more recently, also applied to engineering, biotechnology, marketing, and advertising. The core idea is to design products and processes that are robust to various uncontrollable noise factors.

### Philosophy: Robust Design, Minimizing Variation

*   **Robust Design:** The primary goal of Taguchi methods is to create "robust" products and processes. A robust design is one that performs consistently and reliably, even when exposed to uncontrollable variations (noise factors) in the manufacturing environment, component materials, or operating conditions. Instead of trying to eliminate all sources of variation (which can be expensive or impossible), Taguchi focuses on making the design insensitive to these variations.
*   **Minimizing Variation:** Taguchi emphasized that quality is related to the "loss imparted by the product to society from the time the product is shipped." This loss includes not only manufacturing costs but also costs associated with poor performance, customer dissatisfaction, and environmental impact. He defined quality as the consistency of performance and aimed to minimize variation around a target value. The less variation a product exhibits from its target performance, the higher its quality.
*   **Quality Loss Function (QLF):** Taguchi introduced the concept of the Quality Loss Function, which quantifies the societal loss due to deviation from the target performance. Typically, this is represented as a quadratic function, implying that loss increases as the performance deviates further from the target, even within specification limits.

### Key Contributions by Genichi Taguchi

Dr. Genichi Taguchi (1924-2012) was a Japanese engineer and statistician who made significant contributions to industrial quality control.
*   **Robust Parameter Design:** His most notable contribution is the development of "parameter design," which aims to find the optimal settings of design parameters (control factors) such that the product or process is least sensitive to noise factors.
*   **Orthogonal Arrays:** He promoted the use of Orthogonal Arrays (OAs) for designing experiments efficiently, allowing the study of many factors with a small number of experimental runs.
*   **Signal-to-Noise Ratios:** He introduced Signal-to-Noise (S/N) ratios as objective performance metrics to evaluate the stability and performance of a design.
*   **Off-line Quality Control:** Taguchi advocated for "off-line" quality control, meaning building quality into the product at the design stage, rather than relying on inspection and rework ("on-line" quality control) during production.

## 2. Orthogonal Arrays (OAs)

Orthogonal Arrays are a special type of fractional factorial experimental design that allows for a balanced and efficient study of multiple factors and their main effects. They are a cornerstone of Taguchi's parameter design methodology.

### Concept and Purpose

*   **Balance and Orthogonality:** OAs are balanced in the sense that each level of a factor occurs an equal number of times. They are orthogonal because the estimate of the effect of one factor is not influenced (confounded) by the estimates of the effects of other factors, at least for main effects. This property simplifies the analysis as it allows main effects to be assessed independently.
*   **Efficiency:** OAs allow for the simultaneous study of several factors with a minimal number of experimental runs. For example, an L8 array can study up to 7 factors, each at 2 levels, in just 8 runs. A full factorial design would require $2^7 = 128$ runs.
*   **Focus on Main Effects:** Taguchi methods, particularly when using OAs, primarily focus on identifying the main effects of factors. Interactions are often assumed to be negligible or are handled by specific strategies (e.g., studying them if main effects are very strong, or using larger arrays if certain interactions are critical).

### How to Select an Appropriate OA

The selection of an OA depends on:
1.  **Number of factors** to be investigated.
2.  **Number of levels** for each factor.
3.  The desired **resolution** or the interactions that need to be considered (though Taguchi's standard OAs primarily focus on main effects).

Common OAs are designated by $L_N (l^k)$, where:
*   $N$ is the number of experimental runs (rows in the array).
*   $l$ is the number of levels for each factor (typically 2 or 3).
*   $k$ is the maximum number of factors (columns in the array) that can be accommodated.

**Examples:**
*   **L4 ($2^3$):** 4 runs, can study up to 3 factors, each at 2 levels.
    ```
    Run | A | B | C
    ----|---|---|---
    1   | 1 | 1 | 1
    2   | 1 | 2 | 2
    3   | 2 | 1 | 2
    4   | 2 | 2 | 1
    ```
*   **L8 ($2^7$):** 8 runs, can study up to 7 factors, each at 2 levels.
*   **L9 ($3^4$):** 9 runs, can study up to 4 factors, each at 3 levels.
*   **L18 ($2^1 \times 3^7$):** 18 runs, can study one 2-level factor and up to seven 3-level factors (a mixed-level design).

To select an OA:
1.  List all factors and their chosen levels.
2.  Calculate the total degrees of freedom (DOF) required. For a factor with 'm' levels, the DOF is (m-1). The total DOF for main effects is the sum of DOFs for all factors.
3.  The number of runs (N) in the OA must be greater than or equal to the total DOF required.
4.  Choose the smallest OA that can accommodate all factors and their levels. Standard tables of OAs are widely available.

### Example of a Simple OA Structure (L4)

An L4 array is used to study up to three 2-level factors (A, B, C). The levels are typically coded as 1 (low) and 2 (high).

| Experiment No. | Factor A | Factor B | Factor C | Outcome |
|----------------|----------|----------|----------|---------|
| 1              | 1        | 1        | 1        | $Y_1$   |
| 2              | 1        | 2        | 2        | $Y_2$   |
| 3              | 2        | 1        | 2        | $Y_3$   |
| 4              | 2        | 2        | 1        | $Y_4$   |

*   **Column 1 (Factor A):** 1, 1, 2, 2
*   **Column 2 (Factor B):** 1, 2, 1, 2
*   **Column 3 (Factor C):** 1, 2, 2, 1 (This column can be generated as A x B interaction if only 2 factors are studied, or used for a third factor C if interactions are ignored).

The L4 array allows estimation of the main effects of A, B, and C, assuming interactions are negligible.

## 3. Signal-to-Noise (S/N) Ratios

A key element of Taguchi's methodology is the use of Signal-to-Noise (S/N) ratios to measure quality. The S/N ratio is a performance metric that consolidates several repetitions (or noise conditions) into a single value representing both the average performance (signal) and the variation (noise). The goal is to choose factor levels that maximize the S/N ratio, leading to a design that is robust against noise.

### Explanation of S/N Ratios as Performance Metrics

*   **Signal:** Represents the desired target value or the intended output of the system (mean response).
*   **Noise:** Represents the undesirable variation in the output due to uncontrollable factors (variance or standard deviation).
*   **S/N Ratio:** A measure of robustness. A higher S/N ratio indicates a more robust design, meaning the "signal" (desired output) is strong relative to the "noise" (undesired variability). The S/N ratio is typically expressed in decibels (dB).

The objective in parameter design is to select factor levels that maximize the appropriate S/N ratio.

### Common Types

Taguchi proposed different S/N ratios depending on the desired characteristic of the performance:

1.  **Larger-the-Better (LTB):** Used for characteristics where the ideal value is as large as possible (e.g., strength, yield, efficiency).
    $S/N_{LTB} = -10 \log_{10} \left( \frac{1}{n} \sum_{i=1}^{n} \frac{1}{y_i^2} \right)$
    where $y_i$ is the response value for the i-th trial, and n is the number of repetitions.

2.  **Smaller-the-Better (STB):** Used for characteristics where the ideal value is as small as possible (e.g., wear, shrinkage, defects, pollution).
    $S/N_{STB} = -10 \log_{10} \left( \frac{1}{n} \sum_{i=1}^{n} y_i^2 \right)$

3.  **Nominal-the-Best (NTB):** Used for characteristics where the ideal value is a specific target value, and deviations in either direction are undesirable (e.g., dimension, viscosity, clearance).
    There are two main types for NTB:
    *   **NTB Type I (Variance only):** Used when the mean can be easily adjusted to the target without affecting variability. The goal is to minimize variance.
        $S/N_{NTB_I} = -10 \log_{10} (s^2)$
        where $s^2$ is the sample variance.
    *   **NTB Type II (Mean and Variance):** Used when the mean needs to be at the target and variance needs to be minimized simultaneously. This is the more common NTB.
        $S/N_{NTB_{II}} = -10 \log_{10} \left( \frac{\bar{y}^2}{s^2} \right)$  (if target is non-zero)
        or more generally: $S/N_{NTB_{II}} = 10 \log_{10} \left( \frac{\text{Mean}^2}{\text{Variance}} \right) = 10 \log_{10} \left( \frac{\bar{y}^2}{s^2} \right)$
        (Note: Some formulations use $s^2/\bar{y}^2$ inside the log, leading to a negative sign. The key is consistency: higher is better.)
        If the target is $m$, then $S/N_{NTB} = -10 \log_{10} \left( \sum_{i=1}^{n} \frac{(y_i - m)^2}{n} \right)$ if the mean cannot be easily adjusted. Or if it can, a common form is $S/N = 10 \log (\bar{y}^2 / s^2)$.

### How They Are Used to Identify Optimal Factor Levels

1.  **Calculate S/N for Each Run:** For each experimental run in the OA, conduct multiple repetitions (if noise factors are being simulated) or use the single observed value (if noise is inherent). Calculate the appropriate S/N ratio for that run.
2.  **Average S/N for Each Factor Level:** For each factor, calculate the average S/N ratio for each of its levels. For example, for factor A at level 1, average the S/N ratios from all runs where A was at level 1.
3.  **Select Optimal Levels:** For each factor, choose the level that yields the highest average S/N ratio. This combination of factor levels represents the optimal setting for robustness.
4.  **Response Plots:** Plotting the average S/N ratio for each level of each factor (main effects plots for S/N ratios) helps visualize the impact of each factor and identify the optimal level.

## 4. Steps in a Taguchi Experiment

A typical Taguchi experiment follows a structured approach:

1.  **Define the Problem and Objective:** Clearly state the problem and what needs to be optimized (e.g., improve yield, reduce defects, increase product life). Identify the response variable to be measured.
2.  **Identify Factors and Levels:**
    *   **Control Factors:** Select the design parameters (factors) that can be controlled and their respective levels.
    *   **Noise Factors (Optional but Recommended):** Identify potential noise factors and their levels if a robust design is the primary goal. These can be incorporated into an "outer array" or simulated.
3.  **Select an Appropriate Orthogonal Array (OA):** Choose an OA based on the number of control factors and their levels.
4.  **Design the Experiment Layout:** Assign factors to the columns of the OA. If noise factors are used, design the outer array.
5.  **Conduct the Experiment:** Perform the experimental runs as specified by the OA. Collect data for the response variable(s). If an outer array for noise factors is used, each run of the inner OA (control factors) will be repeated for all conditions of the outer OA.
6.  **Analyze Data and S/N Ratios:**
    *   Calculate the appropriate S/N ratio for each experimental run.
    *   Calculate the average S/N ratio for each factor level.
    *   Determine the optimal level for each factor (the level with the highest average S/N ratio).
    *   Statistical analysis (like ANOVA) can be performed on the S/N ratios to identify statistically significant factors.
7.  **Predict Performance at Optimal Settings:** Predict the S/N ratio and mean response at the determined optimal factor levels.
8.  **Conduct Confirmation Experiment:** Run a confirmation experiment using the optimal factor levels to verify the predicted improvement in performance and robustness.
9.  **Implement and Monitor:** If the confirmation experiment is successful, implement the optimal settings in the process or product design.

## 5. Simple Example

Let's consider a simplified example: **Optimizing Cake Baking** to achieve a consistent height (Nominal-the-Best, assuming a target height).

1.  **Objective:** Bake a cake with a consistent height of 5 cm.
2.  **Factors and Levels (Control Factors):**
    *   Factor A: Oven Temperature (Level 1: 170°C, Level 2: 190°C)
    *   Factor B: Baking Time (Level 1: 30 min, Level 2: 40 min)
    *   Factor C: Amount of Flour (Level 1: 200g, Level 2: 220g)
3.  **Select OA:** 3 factors, 2 levels each. We can use an L4($2^3$) array.

    | Run | Temp (A) | Time (B) | Flour (C) | Height (y1) | Height (y2) | S/N (NTB) |
    |-----|----------|----------|-----------|-------------|-------------|-----------|
    | 1   | 1 (170°) | 1 (30m)  | 1 (200g)  | 4.8 cm      | 4.9 cm      | $S/N_1$   |
    | 2   | 1 (170°) | 2 (40m)  | 2 (220g)  | 5.1 cm      | 5.2 cm      | $S/N_2$   |
    | 3   | 2 (190°) | 1 (30m)  | 2 (220g)  | 5.3 cm      | 5.4 cm      | $S/N_3$   |
    | 4   | 2 (190°) | 2 (40m)  | 1 (200g)  | 4.7 cm      | 4.6 cm      | $S/N_4$   |
    *(Assume two repetitions (y1, y2) for each run to simulate noise/variability)*

4.  **Conduct Experiment & Collect Data:** Bake cakes according to the L4 design and measure heights (as shown above).
5.  **Analyze S/N Ratios:**
    *   For each run, calculate the S/N ratio. Let's use Nominal-the-Best Type II: $S/N = 10 \log_{10} (\bar{y}^2 / s^2)$.
        *   Example for Run 1: $\bar{y}_1 = (4.8+4.9)/2 = 4.85$. $s_1^2 = ((4.8-4.85)^2 + (4.9-4.85)^2)/(2-1) = 0.005$.
          $S/N_1 = 10 \log_{10} (4.85^2 / 0.005) \approx 36.72$ dB.
        *   Similarly, calculate $S/N_2, S/N_3, S/N_4$. (Actual values would depend on the data).
6.  **Average S/N for Each Factor Level:**
    *   **Temp (A):**
        *   Level 1 (170°C): Average($S/N_1, S/N_2$)
        *   Level 2 (190°C): Average($S/N_3, S/N_4$)
    *   **Time (B):**
        *   Level 1 (30 min): Average($S/N_1, S/N_3$)
        *   Level 2 (40 min): Average($S/N_2, S/N_4$)
    *   **Flour (C):**
        *   Level 1 (200g): Average($S/N_1, S/N_4$)
        *   Level 2 (220g): Average($S/N_2, S/N_3$)
7.  **Select Optimal Levels:** Choose the level for each factor that gives the highest average S/N ratio. For instance, if A2, B1, C2 show the highest S/N averages, then the optimal setting is Temp=190°C, Time=30min, Flour=220g.
8.  **Confirmation Experiment:** Bake a cake using these optimal settings and check if the height is consistently close to 5cm.

This example illustrates how Taguchi methods structure experimentation to find robust settings efficiently. The focus is on achieving consistent performance (desired height) by minimizing sensitivity to variations.

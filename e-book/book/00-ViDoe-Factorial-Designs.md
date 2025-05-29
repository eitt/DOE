<a name="00-ViDoe-Factorial-Designs"></a>
# Factorial Analysis and Sensibility
```{contents}
```

## Introduction to Factorial Designs

Factorial designs are a powerful class of experimental designs that allow researchers to study the effects of multiple factors (independent variables) on a response variable simultaneously. Instead of studying one factor at a time, factorial designs enable the examination of not only the main effect of each factor but also the interaction effects between factors. This means we can understand how the effect of one factor might change depending on the level of another factor.

These designs are widely used in various fields, including engineering, manufacturing, agriculture, and social sciences, to optimize processes, improve product quality, and identify significant factors influencing an outcome.

## $2^k$ Factorial Designs

A $2^k$ factorial design is a specific type of factorial design where 'k' represents the number of factors being investigated, and each of these factors has exactly two levels. These levels are typically qualitative (e.g., "low" vs. "high", "type A" vs. "type B") or quantitative (e.g., 10°C vs. 20°C, 50rpm vs. 100rpm).

### Concept

*   **k factors:** The design involves 'k' distinct independent variables that are believed to influence the response.
*   **2 levels:** Each factor is studied at two specific levels. These are often denoted as '-' and '+' or -1 and +1 for the low and high levels, respectively.
*   **Full Factorial:** A $2^k$ design is a "full factorial" design because it includes all possible combinations of the levels of the k factors. The total number of experimental runs required is $2^k$. For example:
    *   A $2^2$ design has 2 factors, each at 2 levels, resulting in $2^2 = 4$ runs.
    *   A $2^3$ design has 3 factors, each at 2 levels, resulting in $2^3 = 8$ runs.

### Setting up $2^k$ Designs

Setting up a $2^k$ design involves defining the factors, their levels, and the standard order for the experimental runs.

1.  **Identify Factors and Levels:** Clearly define the k factors and the specific low (-) and high (+) levels for each.
2.  **Standard Order:** A standard order, often called "Yates order," is used to list the treatment combinations. For a $2^3$ design with factors A, B, and C, the standard order is:

    | Run | A   | B   | C   | Combination |
    |-----|-----|-----|-----|-------------|
    | 1   | -   | -   | -   | (1)         |
    | 2   | +   | -   | -   | a           |
    | 3   | -   | +   | -   | b           |
    | 4   | +   | +   | -   | ab          |
    | 5   | -   | -   | +   | c           |
    | 6   | +   | -   | +   | ac          |
    | 7   | -   | +   | +   | bc          |
    | 8   | +   | +   | +   | abc         |

    The pattern for generating this table is:
    *   Factor A alternates levels every run (-, +, -, +, ...).
    *   Factor B alternates levels every two runs (-, -, +, +, ...).
    *   Factor C alternates levels every four runs (-, -, -, -, +, +, +, +).
    *   This pattern continues for k factors, with the j-th factor alternating levels every $2^{j-1}$ runs.

### Analysis of Main Effects and Interactions

After conducting the experiments and collecting the response data for each run, the analysis focuses on:

*   **Main Effects:** The main effect of a factor is the average change in the response variable produced by a change in the level of that factor, averaged over the levels of the other factors. For factor A, it's calculated as:
    $Effect_A = \bar{Y}_{A+} - \bar{Y}_{A-}$
    where $\bar{Y}_{A+}$ is the average response when A is at its high level, and $\bar{Y}_{A-}$ is the average response when A is at its low level.

*   **Interaction Effects:** An interaction effect occurs when the effect of one factor on the response depends on the level of another factor. A two-factor interaction (e.g., AB) is the difference between the effect of A at the high level of B and the effect of A at the low level of B.
    $Effect_{AB} = \frac{[(abc) + (ab) + (c) + (1)]_{avg_B_+} - [(ac) + (a) + (bc) + (b)]_{avg_B_-}}{2}$ (this is one way to calculate it, more complex expressions exist)
    More generally, the effect of an interaction (like AB) is calculated by taking the combinations where A and B are at the same level ((+) with (+) or (-) with (-)) minus the combinations where they are at different levels ((+) with (-) or (-) with (+)), and dividing by $2^{k-1}$.

    For example, the AB interaction effect in a $2^3$ design:
    $Effect_{AB} = \frac{1}{4} [ (abc) - (bc) - (ac) + (c) + (ab) - (b) - (a) + (1) ]$
    where (abc), (bc), etc., represent the response values for those treatment combinations.

Statistical methods like Analysis of Variance (ANOVA) are commonly used to determine the statistical significance of these effects.

### Simple Example: $2^2$ Design

Let's say we want to study the effect of **Temperature (A)** and **Pressure (B)** on the **Yield (Y)** of a chemical process.
*   Factor A (Temperature): Low (-) = 100°C, High (+) = 150°C
*   Factor B (Pressure): Low (-) = 50 psi, High (+) = 75 psi

The $2^2 = 4$ experimental runs are:

| Run | A (Temp) | B (Press) | Yield (Y) |
|-----|----------|-----------|-----------|
| 1   | - (100°C)| - (50psi) | $Y_1$     |
| 2   | + (150°C)| - (50psi) | $Y_2$     |
| 3   | - (100°C)| + (75psi) | $Y_3$     |
| 4   | + (150°C)| + (75psi) | $Y_4$     |

**Effects Calculation:**
*   $Effect_A = \frac{(Y_2 + Y_4)}{2} - \frac{(Y_1 + Y_3)}{2} = \frac{1}{2} (Y_2 - Y_1 - Y_3 + Y_4)$
*   $Effect_B = \frac{(Y_3 + Y_4)}{2} - \frac{(Y_1 + Y_2)}{2} = \frac{1}{2} (Y_3 - Y_1 - Y_2 + Y_4)$
*   $Effect_{AB} = \frac{(Y_1 + Y_4)}{2} - \frac{(Y_2 + Y_3)}{2} = \frac{1}{2} (Y_1 - Y_2 - Y_3 + Y_4)$

If $Effect_A$ is large and positive, it means increasing temperature from 100°C to 150°C increases yield. If $Effect_{AB}$ is non-zero, it means the effect of temperature on yield depends on the pressure level (and vice-versa).

## $2^{k-p}$ Fractional Factorial Designs

As the number of factors (k) in a $2^k$ design increases, the number of required experimental runs ($2^k$) grows very rapidly. For example, a $2^5$ design requires 32 runs, and a $2^7$ design requires 128 runs. In many practical situations, running such a large number of experiments can be too expensive, time-consuming, or resource-intensive.

$2^{k-p}$ fractional factorial designs offer a solution by allowing researchers to study k factors in fewer than $2^k$ runs. These designs use a carefully chosen fraction (specifically, a $1/2^p$ fraction) of the full factorial design.

### Concept

*   **k factors, 2 levels:** Similar to full factorial designs, each of the k factors is studied at two levels.
*   **Fraction of a full design:** A $2^{k-p}$ design consists of $2^{k-p}$ runs, which is a $1/2^p$ fraction of the $2^k$ full factorial design.
    *   'p' is the number that determines the fraction size.
        *   If p=1, it's a half-fraction design ($2^{k-1}$).
        *   If p=2, it's a quarter-fraction design ($2^{k-2}$).
*   **Resource Saving:** The primary motivation for using fractional factorial designs is to reduce the number of experimental runs, thereby saving time, cost, and materials. This is particularly useful in screening experiments where the goal is to identify the most important factors out of many potential candidates.

### Why Use Fractional Factorial Designs?

*   **Efficiency:** Fewer runs are needed compared to a full factorial design.
*   **Screening:** Excellent for identifying the vital few factors from the trivial many, especially when k is large.
*   **Sparsity of Effects Principle:** Often, in systems with many factors, only a few main effects and low-order interactions are significant. Higher-order interactions (e.g., three-factor or higher) are often negligible. Fractional designs exploit this principle.

### Design Resolution and Aliasing

The main drawback of fractional factorial designs is that not all effects can be estimated independently. Some effects become "aliased" or "confounded" with other effects, meaning their estimates are combined and cannot be separated.

*   **Alias Structure:** The alias structure describes which effects are confounded with each other. It's determined by the "defining relation" or "design generator(s)" used to construct the fractional design.
    For example, in a $2^{3-1}$ design (3 factors in $2^{3-1}=4$ runs), we might choose the defining relation $I = ABC$. This means the main effect of A is aliased with the BC interaction, B with AC, and C with AB.
    *   $L_A = A + BC$
    *   $L_B = B + AC$
    *   $L_C = C + AB$
    (where $L_A$ is the linear combination estimated for A, etc.)

*   **Design Resolution:** The resolution of a fractional factorial design indicates the extent of aliasing. It is denoted by a Roman numeral (e.g., Resolution III, IV, V).
    *   **Resolution III Designs:** Main effects are aliased with two-factor interactions (e.g., $A = A + BC$). These designs are useful for screening when two-factor interactions are assumed to be negligible. (Generator example: $I=ABC$ for $2^{3-1}$)
    *   **Resolution IV Designs:** No main effects are aliased with two-factor interactions, but two-factor interactions are aliased with other two-factor interactions (e.g., $AB = AB + CD$). Main effects are aliased with three-factor interactions. These are good for situations where main effects are of primary interest and some two-factor interactions might be important. (Generator example: $I=ABCD$ for $2^{4-1}$)
    *   **Resolution V Designs:** No main effects are aliased with two-factor or three-factor interactions. No two-factor interactions are aliased with other two-factor interactions. Two-factor interactions are aliased with three-factor interactions. These designs allow estimation of all main effects and two-factor interactions, assuming three-factor and higher interactions are negligible. (Generator example: $I=ABCDE$ for $2^{5-1}$)

    Higher resolution is generally better but requires more runs. The choice of resolution depends on the experimental objectives and prior knowledge about the system.

### Simple Example: $2^{3-1}$ Design (Resolution III)

Suppose we have 3 factors (A, B, C) but can only afford 4 runs instead of $2^3 = 8$. We can construct a $2^{3-1}$ design.

1.  **Base Design:** Start with a full $2^{3-1} = 2^2$ design for factors A and B.

    | Run | A   | B   |
    |-----|-----|-----|
    | 1   | -   | -   |
    | 2   | +   | -   |
    | 3   | -   | +   |
    | 4   | +   | +   |

2.  **Generate Level for Factor C:** We need to define how to set the levels for factor C. We can do this by setting C equal to an interaction of the base factors, for example, $C = AB$ or $C = -AB$. Let's choose $C = AB$. This means the defining relation is $I = ABC$ (because if $C=AB$, then $C \cdot C = AB \cdot C \implies I = ABC$).

    The design becomes:

    | Run | A   | B   | C = AB |
    |-----|-----|-----|--------|
    | 1   | -   | -   | +      |  (-)(-)=+
    | 2   | +   | -   | -      |  (+)(-)=-
    | 3   | -   | +   | -      |  (-)(+)=-
    | 4   | +   | +   | +      |  (+)(+)=+

    This is one half of the full $2^3$ design (specifically, the "principal fraction" if $I = +ABC$).

3.  **Alias Structure (with $I=ABC$):**
    *   $A$ is aliased with $BC$ ($A \equiv A \cdot ABC = A^2BC \equiv BC$)
    *   $B$ is aliased with $AC$
    *   $C$ is aliased with $AB$

    When we estimate the main effect of A, we are actually estimating $A+BC$. If the $BC$ interaction is significant, our estimate for A will be biased. This is the trade-off for reducing the number of runs.

Fractional factorial designs are a cornerstone of efficient experimentation, especially in the initial stages of an investigation or when resources are limited. Careful selection of the design and awareness of its alias structure are crucial for drawing valid conclusions.

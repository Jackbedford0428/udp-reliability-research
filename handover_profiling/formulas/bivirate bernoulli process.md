Bivariate Bernoulli Process
===

# Individual Probability Distribution

$ X \sim Bernoulli(p) \Rightarrow P(X=1) = p, 0 \le p \le 1 \\ $
$ Y \sim Bernoulli(q) \Rightarrow P(Y=1) = q, 0 \le q \le 1 \\ $

# Correlation Coefficient

$ \rho_{XY} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}, -1 \le \rho_{XY} \le 1 \\ $

# Joint Probabilities

$ a := P(X=0, Y=0) \\ $
$ b := P(X=0, Y=1) = (1-p) - a \\ $
$ c := P(X=1, Y=0) = (1-q) - a \\ $
$ d := P(X=1, Y=1) = 1 - a - b - c = a + p + q - 1 \\ $
\
$ p = c + d \\ $
$ q = b + d \\ $
$ a + b + c + d = 1 \\ $

# Conditional Probabilities

$ \alpha := P(Y=1|X=1) = \frac{P(X=1, Y=1)}{P(X=1)} = \frac{d}{p} \\ $
$ \beta := P(Y=1|X=0) = \frac{P(X=0, Y=1)}{P(X=0)} = \frac{b}{1-p} \\ $
$ \gamma := P(X=1|Y=1) = \frac{P(X=1, Y=1)}{P(Y=1)} = \frac{d}{q} \\ $
$ \delta := P(X=1|Y=0) = \frac{P(X=1, Y=0)}{P(Y=0)} = \frac{c}{1-q} \\ $

# Derivations

$ \rho_{XY} = \frac{Cov(X, Y)}{\sigma_X \sigma_Y} \\ $

## Expected Value

$$
\begin{align}
E[X] &= \sum_{x \in \{ 0, 1 \}} xP(X=x) \\
&= 0 \cdot (1-p) + 1 \cdot p = p \\
\end{align}
$$

$$
\begin{align} 
E[Y] &= \sum_{y \in \{ 0, 1 \}} yP(Y=y) \\
&= 0 \cdot (1-q) + 1 \cdot q = q \\
\end{align}
$$

## Variance

$$
\begin{align} 
Var(X) &= E[(X - E[X])^2] = E[X^2] - E[X]^2 \\
&= \sum_{x \in \{ 0, 1 \}} x^2 P(X=x) - p^2 \\
&= 0 \cdot (1-p) + 1 \cdot p - p^2 = p - p^2 \\
&= p(1-p) \\
\end{align}
$$

$$
\begin{align} 
Var(Y) &= E[(Y - E[Y])^2] = E[Y^2] - E[Y]^2 \\
&= \sum_{y \in \{ 0, 1 \}} y^2 P(Y=y) - q^2 \\
&= 0 \cdot (1-q) + 1 \cdot q - q^2 = q - q^2 \\
&= q(1-q) \\
\end{align}
$$

## Covariance

$$
\begin{align} 
Cov(X, Y) &= E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y] \\
&= \sum_{x \in \{ 0, 1 \}} \sum_{y \in \{ 0, 1 \}} xy P(X=x, Y=y) - pq \\
&= \sum_{x \in \{ 0, 1 \}} x \cdot 0 \cdot P(X=x, Y=0) + x \cdot 1 \cdot P(X=x, Y=1) - pq \\
&= 0 \cdot P(X=0, Y=1) + 1 \cdot P(X=1, Y=1) - pq \\
&= d - pq = a + p + q - 1 - pq \\
&= a - (1-p)(1-q) \\
\end{align}
$$

## Correlation Coefficient

$$
\begin{align} 
\rho_{XY} &= \frac{Cov(X, Y)}{\sigma_X \sigma_Y} \\
&= \frac{d - pq}{\sigma_X \sigma_Y} = \frac{p \alpha - pq}{\sigma_X \sigma_Y} = \frac{q \gamma - pq}{\sigma_X \sigma_Y} \\
&= \frac{a - (1-p)(1-q)}{\sigma_X \sigma_Y} \\
\end{align}
$$

$$
\begin{align} \sigma_X \sigma_Y &= \sqrt{Var(X) \cdot Var(Y)} = \sqrt{pq(1-p)(1-q)} \\
\end{align}
$$

# Given $p$, $q$, $\rho$

$ a := P(X=0, Y=0) = (1-p)(1-q) + \rho \sigma_X \sigma_Y \\ $
$ b := P(X=0, Y=1) = (1-p) - a = q(1-p) - \rho \sigma_X \sigma_Y = (1-p)\beta \\ $
$ c := P(X=1, Y=0) = (1-q) - a = p(1-q) - \rho \sigma_X \sigma_Y = (1-q)\delta \\ $
$ d := P(X=1, Y=1) = pq + \rho \sigma_X \sigma_Y = p\alpha = q\gamma \\ $
\
$ \Rightarrow \\ $
$ \alpha := P(Y=1|X=1) = \frac{P(X=1, Y=1)}{P(X=1)} = \frac{d}{p} = \frac{pq + \rho \sigma_X \sigma_Y}{p} \\ $
$ \beta := P(Y=1|X=0) = \frac{P(X=0, Y=1)}{P(X=0)} = \frac{b}{1-p} = \frac{q(1-p) - \rho \sigma_X \sigma_Y}{1-p} \\ $
$ \gamma := P(X=1|Y=1) = \frac{P(X=1, Y=1)}{P(Y=1)} = \frac{d}{q} = \frac{pq + \rho \sigma_X \sigma_Y}{q} \\ $
$ \delta := P(X=1|Y=0) = \frac{P(X=1, Y=0)}{P(Y=0)} = \frac{c}{1-q} = \frac{p(1-q) - \rho \sigma_X \sigma_Y}{1-q} \\ $

# Restrictions of $\rho$

$$
\begin{align} -1 \leq \rho \leq 1 \\
\end{align}
$$

$$
\begin{align} 0 \leq a \leq 1 \Rightarrow
-\frac{(1-p)(1-q)}{\sigma_X \sigma_Y} \leq \rho \leq \frac{1-(1-p)(1-q)}{\sigma_X \sigma_Y} \\
\end{align}
$$

$$
\begin{align} 0 \leq b \leq 1 \Rightarrow
\frac{q(1-p)-1}{\sigma_X \sigma_Y} \leq \rho \leq \frac{q(1-p)}{\sigma_X \sigma_Y} \\
\end{align}
$$

$$
\begin{align} 0 \leq c \leq 1 \Rightarrow
\frac{p(1-q)-1}{\sigma_X \sigma_Y} \leq \rho \leq \frac{p(1-q)}{\sigma_X \sigma_Y} \\
\end{align}
$$

$$
\begin{align} 0 \leq d \leq 1 \Rightarrow
-\frac{pq}{\sigma_X \sigma_Y} \leq \rho \leq \frac{1-pq}{\sigma_X \sigma_Y} \\
\end{align}
$$

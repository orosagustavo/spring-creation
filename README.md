# Particle Creation in Fortran
Project designed to test the creation of particles connected by a spring.

## First Version - Damped Harmonic Oscillator
To begin, we need to study the equations of an oscillator. For this, we use Newton's second law:

$$m \frac{d^{2} \vec{r}'}{dt^{2}} = -k \vec{r}' + b \frac{d \vec{r}'}{dt}$$

where:
- $m$ is the particle's mass
- $k$ is the spring constant
- $b$ is the damping constant

This would be great for the case of a particle connected to a fixed point. However, we're aiming to expand this to two particles connected to each other. For that, we've defined the vector $\vec{r'}$ as a relative vector between the two particles, given by:

$$\vec{r}' = (d - L) * \frac{x^{2} - x^{1}}{|x^{2} - x^{1}|}$$

where:
- $d$ is the relative distance between the particles
- $L$ is the spring's natural length

### Numerical Calculation of the Equations
With the differential equation defined, we reduce it to first order by considering $\frac{d \vec{r}'}{dt} = v$. Using this, we can directly compute the acceleration of each particle with Newton's second law:

$$a_i^{(n)} = \frac{F_i^{(n)} - F_{\text{damp}, i}^{(n)}}{m^{(n)}}$$


where:
- $n$ corresponds to the particle
- $i$ corresponds to the vector component $x$ or $y$

With the acceleration in hand, we can compute velocity using the [Verlet Integration Method](https://en.wikipedia.org/wiki/Verlet_integration):

$$\vec{v}(t + dt) = \vec{v}(t) + \frac{1}{2}[\vec{a}(t+dt) + \vec{a}(t)] dt$$

and position as well:

$$\vec{r}(t+dt) = \vec{r}(t) + \vec{v}(t+dt) dt + \frac{1}{2} \vec{a}(t+dt) dt^{2}$$

### Program Output
All output is in `.xyz` format to be visualized in a program like `Ovito`.

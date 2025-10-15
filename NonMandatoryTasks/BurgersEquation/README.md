# Burgers' 2D equations

$$
\begin{aligned}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} &= \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} &= \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right),
\end{aligned}
$$

where:
- $u(x, y, t)$ and $v(x, y, t)$ are the velocity components in the $x$ and $y$ directions,
- $\nu$ is the kinematic viscosity,
- $t$ is time.

These are the 2D Burgers' equations, derived from the Navier–Stokes equations under the following assumptions:
## Assumptions to Derive the 2D Burgers' Equation from Navier–Stokes

1. **Neglect pressure gradient**
   The pressure term $\nabla p$ is dropped to focus purely on convection and diffusion effects. This simplifies the equations by removing the need to solve for pressure.

2. **No external forces**
   External body forces like gravity are ignored, further simplifying the model.

3. **Relaxed incompressibility**
   In some versions of the Burgers' equation, the incompressibility condition
   $\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$
   is dropped to simplify the analysis or numerical solution.

4. **Simplified 2D velocity field**
   Both velocity components $u(x, y, t)$ and $v(x, y, t)$ are retained, but the flow is treated in a simplified 2D framework without pressure coupling.

## Concept of ML
You don't need to solve those equations; don't worry, just knowing how to do differentiation is sufficient for this task. In this task you need to explore the concept of physics in deep learning. It's fine if you dislike physics — for now just take this as math.

Explore the concepts of PINNs, create your own dataset (synthetic), train a PINN, optimize the loss function, and predict.
Once you have done those, you might think this is unnecessary, but if you feel that way then you did something wrong. Create a small training dataset — you can still achieve great accuracy.

## Task Instruction
### Steps:

- Create a synthetic Burgers' 2D dataset with varying $\nu$; it should be time series as well.
- Visualize them and explain how the patterns look.
- Create a grouped split for train and test — i.e., the train dataset and test dataset should not have anything in common, even $\nu$ (this is the only way you'll see the strengths of PINNs).
- Visualize and compare true and predicted values.
- Try adding noise and see how well the model handles it, and which hyperparameters need to be changed to get better results with a lot of noise.

## Note

The task you will do is far from reality, as this is a synthetic dataset. I would greatly appreciate it if you could find a real dataset (not only Burgers' equation) with real-world scenarios or physics solver data points and train your PINN on it. 🥲

There are many other complex CFD problems out there. They fall into three major categories:
1. Heat transfer
2. Mass transfer
3. Fluid flow

These are where PINNs are mostly used. But PINNs can be used anywhere — for example, when there is a lot of noise in data that should follow physics, or when the governing PDEs are very complex to solve.

I suggest you look into fluid flow with turbulence, which is one of the most common problem statements for PINNs (we don't expect you to solve it here; just explore it): [link1](https://turbulence.idies.jhu.edu/home) — this website provides databases for fluid flow.

[link2](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/) — NVIDIA's site, which contains extensive material on physics-based models, datasets, metrics, custom losses, examples, and pipelines. It's very well maintained.

Hope this task opens a new niche in data science for you.

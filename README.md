# Danny Cloth Simulator

This project is based on the `taichi` library, using a mass-spring system to simulate cloth.

Implemented three methods, which are:

1. Jacobi Method
2. Gauss-Serdel Method
3. Conjugate gradient method



## Result

> The result of each method generated when the number of particles is 16*16.

1. Jacobi Method

   <img src="./README.assets/Jacobi.gif" alt="Jacobi" style="zoom:50%;" />

2. Gauss-Serdel Method

   <img src="./assets/GS.gif" alt="GS" style="zoom:50%;" />

3. Conjugate gradient method

   <img src="./assets/CG.gif" alt="CG" style="zoom:50%;" />

## Run

### Requirement

```
taichi >= 1.7.0
```

### Run with window

```
python main.py
```

### Save 

```
python save.py
```


import numpy as np
import matplotlib.pyplot as plt

# Function to generate velocity profile data (example placeholders)
def generate_velocity_profile(Re):
    z = np.linspace(0, 1, 100)
    u = np.tanh(10 * (z - 0.5)) * (Re / 100)
    return z, u

def generate_velocity_contour(Re):
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    U = np.sin(np.pi * X) * np.cos(np.pi * Y) * (Re / 100)
    V = -np.cos(np.pi * X) * np.sin(np.pi * Y) * (Re / 100)
    return X, Y, U, V

def generate_vorticity_contour(Re):
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.pi * X) * np.sin(np.pi * Y) * (Re / 100)
    return X, Y, Z

# New graphs: Velocity magnitude and streamlines
def generate_velocity_magnitude(Re):
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    U = np.sin(np.pi * X) * np.cos(np.pi * Y) * (Re / 100)
    V = -np.cos(np.pi * X) * np.sin(np.pi * Y) * (Re / 100)
    magnitude = np.sqrt(U*2 + V*2)
    return X, Y, magnitude

def generate_streamlines(Re):
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    U = np.sin(np.pi * X) * np.cos(np.pi * Y) * (Re / 100)
    V = -np.cos(np.pi * X) * np.sin(np.pi * Y) * (Re / 100)
    return X, Y, U, V

# Plot 1: Velocity profile on vertical centerline
Re = 100
z, u = generate_velocity_profile(Re)
plt.figure(figsize=(8, 6))
plt.plot(z, u, label=f'Re={Re}')
plt.title('Velocity Profile on Vertical Centerline')
plt.xlabel('z')
plt.ylabel('u')
plt.legend()
plt.grid()
plt.show()

# Plot 2: Velocity vectors on a plane (y=0.5 as example)
X, Y, U, V = generate_velocity_contour(Re)
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, scale=50)
plt.title('Velocity Vectors at y=0.5 Plane')
plt.xlabel('x')
plt.ylabel('z')
plt.grid()
plt.show()

# Plot 3: Vorticity contours on a plane (y=0.5 as example)
X, Y, Z = generate_vorticity_contour(Re)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, cmap='viridis')
plt.title('Vorticity Contours at y=0.5 Plane')
plt.xlabel('x')
plt.ylabel('z')
plt.colorbar(contour)
plt.grid()
plt.show()

# Plot 4: Velocity magnitude contours
X, Y, magnitude = generate_velocity_magnitude(Re)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, magnitude, cmap='plasma')
plt.title('Velocity Magnitude Contours')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(contour)
plt.grid()
plt.show()

# Plot 5: Streamlines
X, Y, U, V = generate_streamlines(Re)
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, U, V, color=np.sqrt(U*2 + V*2), cmap='cool')
plt.title('Streamlines')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Velocity magnitude')
plt.grid()
plt.show()
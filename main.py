import pygame
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fluid Simulation with customizable parameters')
    
    # Visualization options
    parser.add_argument('--show-arrows', action='store_true', help='Show velocity arrows')
    parser.add_argument('--arrow-spacing', type=int, default=3, help='Spacing between arrows (default: 3)')
    parser.add_argument('--scale', type=int, default=5, help='Grid cell size (default: 5)')
    
    # Physical parameters
    parser.add_argument('--viscosity', type=float, default=0.0000001, help='Fluid viscosity (default: 0.0000001)')
    parser.add_argument('--diffusion', type=float, default=0.0000001, help='Diffusion rate (default: 0.0000001)')
    parser.add_argument('--vorticity', type=float, default=0.1, help='Vorticity strength (default: 0.1)')
    parser.add_argument('--buoyancy', type=float, default=0.1, help='Buoyancy strength (default: 0.1)')
    parser.add_argument('--cooling-rate', type=float, default=0.995, help='Cooling rate (default: 0.995)')
    parser.add_argument('--velocity-damping', type=float, default=0.99, help='Velocity damping (default: 0.99)')
    
    # Display options
    parser.add_argument('--width', type=int, default=1200, help='Window width (default: 1200)')
    parser.add_argument('--height', type=int, default=800, help='Window height (default: 800)')
    
    return parser.parse_args()

class FluidSimulation:
    def __init__(self, width, height, args):
        self.scale = args.scale
        self.N = width // self.scale
        self.iter = 16
        self.show_arrows = args.show_arrows
        self.arrow_spacing = args.arrow_spacing
        
        # Physical parameters from CLI
        self.dt = 0.1
        self.diff = args.diffusion
        self.visc = args.viscosity
        self.vorticity = args.vorticity
        self.buoyancy = args.buoyancy
        self.cooling_rate = args.cooling_rate
        self.velocity_damping = args.velocity_damping
        
        # Initialize arrays
        self.size = (self.N, self.N)
        self.s = np.zeros(self.size)
        self.density = np.zeros(self.size)
        self.temperature = np.zeros(self.size)
        self.Vx = np.zeros(self.size)
        self.Vy = np.zeros(self.size)
        self.Vx0 = np.zeros(self.size)
        self.Vy0 = np.zeros(self.size)


    def add_density(self, x, y, amount, temp=50):
        self.density[x, y] += amount
        self.temperature[x, y] += temp

    def add_velocity(self, x, y, amount_x, amount_y):
        self.Vx[x, y] += amount_x
        self.Vy[x, y] += amount_y

    def apply_buoyancy(self):
        # Apply buoyancy force based on temperature
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                self.Vy[i, j] -= self.buoyancy * self.temperature[i, j]

    def apply_vorticity(self):
        # Calculate and apply vorticity confinement
        curl = np.zeros(self.size)
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                curl[i, j] = (self.Vy[i+1, j] - self.Vy[i-1, j] -
                            self.Vx[i, j+1] + self.Vx[i, j-1]) * 0.5
                
        # Apply force
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                force = curl[i, j] * self.vorticity
                self.Vx[i, j] += force
                self.Vy[i, j] += force

    def diffuse(self, b, x, x0, diff):
        a = self.dt * diff * (self.N - 2) * (self.N - 2)
        self.lin_solve(b, x, x0, a, 1 + 6 * a)
        
    def lin_solve(self, b, x, x0, a, c):
        c_recip = 1.0 / c
        for _ in range(self.iter):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + 
                            a * (x[2:, 1:-1] + 
                                x[:-2, 1:-1] + 
                                x[1:-1, 2:] + 
                                x[1:-1, :-2])) * c_recip
            self.set_bnd(b, x)

    def project(self, velocX, velocY, p, div):
        div[1:-1, 1:-1] = (-0.5 * (
                velocX[2:, 1:-1] - 
                velocX[:-2, 1:-1] + 
                velocY[1:-1, 2:] - 
                velocY[1:-1, :-2])) / self.N
        p *= 0
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        self.lin_solve(0, p, div, 1, 6)
        
        velocX[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.N
        velocY[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.N
        
        self.set_bnd(1, velocX)
        self.set_bnd(2, velocY)

    def advect(self, b, d, d0, velocX, velocY):
        dtx = self.dt * (self.N - 2)
        dty = self.dt * (self.N - 2)

        for j in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                tmp1 = dtx * velocX[i, j]
                tmp2 = dty * velocY[i, j]
                x = i - tmp1
                y = j - tmp2
                
                if x < 0.5: x = 0.5
                if x > self.N - 1.5: x = self.N - 1.5
                i0 = int(x)
                i1 = i0 + 1
                
                if y < 0.5: y = 0.5
                if y > self.N - 1.5: y = self.N - 1.5
                j0 = int(y)
                j1 = j0 + 1
                
                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1
                
                d[i, j] = (s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
                          s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1]))
        
        self.set_bnd(b, d)

    def set_bnd(self, b, x):
        if b == 1:
            x[0, :] = -x[1, :]
            x[-1, :] = -x[-2, :]
        else:
            x[0, :] = x[1, :]
            x[-1, :] = x[-2, :]
            
        if b == 2:
            x[:, 0] = -x[:, 1]
            x[:, -1] = -x[:, -2]
        else:
            x[:, 0] = x[:, 1]
            x[:, -1] = x[:, -2]

    def step(self):
        # Apply enhanced physics
        self.apply_buoyancy()
        self.apply_vorticity()
        
        # Velocity step
        self.diffuse(1, self.Vx0, self.Vx, self.visc)
        self.diffuse(2, self.Vy0, self.Vy, self.visc)
        
        self.project(self.Vx0, self.Vy0, self.Vx, self.Vy)
        
        self.advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0)
        self.advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0)
        
        self.project(self.Vx, self.Vy, self.Vx0, self.Vy0)
        
        # Density step
        self.diffuse(0, self.s, self.density, self.diff)
        self.advect(0, self.density, self.s, self.Vx, self.Vy)
        
        # Apply cooling and damping
        self.density *= self.cooling_rate
        self.temperature *= self.cooling_rate
        self.Vx *= self.velocity_damping
        self.Vy *= self.velocity_damping

    def draw_arrow(self, screen, start_pos, velocity, scale=15):
        if np.any(velocity):
            end_pos = (
                start_pos[0] + velocity[0] * scale,
                start_pos[1] + velocity[1] * scale
            )
            
            pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 1)
            
            if np.linalg.norm(velocity) > 0.1:
                angle = np.arctan2(velocity[1], velocity[0])
                arrow_size = 5
                arrow_angle = np.pi/6
                
                point1 = (
                    end_pos[0] - arrow_size * np.cos(angle + arrow_angle),
                    end_pos[1] - arrow_size * np.sin(angle + arrow_angle)
                )
                point2 = (
                    end_pos[0] - arrow_size * np.cos(angle - arrow_angle),
                    end_pos[1] - arrow_size * np.sin(angle - arrow_angle)
                )
                
                pygame.draw.line(screen, (255, 255, 255), end_pos, point1, 1)
                pygame.draw.line(screen, (255, 255, 255), end_pos, point2, 1)
    def draw(self, screen):
        # Draw density field with temperature influence
        for i in range(self.N):
            for j in range(self.N):
                x = i * self.scale
                y = j * self.scale
                d = self.density[i, j]
                t = self.temperature[i, j]
                green = int(min(d * 255, 255))
                red = int(min(t * 2, 255))
                color = (red, green, 0)
                pygame.draw.rect(screen, color, (x, y, self.scale, self.scale))
        
        # Draw velocity field if enabled
        if self.show_arrows:
            for i in range(0, self.N, self.arrow_spacing):
                for j in range(0, self.N, self.arrow_spacing):
                    x = i * self.scale + self.scale // 2
                    y = j * self.scale + self.scale // 2
                    velocity = np.array([self.Vx[i, j], self.Vy[i, j]])
                    if np.linalg.norm(velocity) > 0.01:
                        self.draw_arrow(screen, (x, y), velocity)

def print_controls():
    print("\nSimulation Controls:")
    print("-------------------")
    print("Space: Reset simulation")
    print("ESC: Exit")
    print("A: Toggle velocity arrows")
    print("Up/Down: Adjust vorticity")
    print("Left/Right: Adjust buoyancy")
    print("B/N: Adjust cooling rate")
    print("V/C: Adjust velocity damping")
    print("\nCurrent parameters can be viewed in the window title")

def main():
    args = parse_arguments()
    
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    clock = pygame.time.Clock()
    
    fluid = FluidSimulation(args.width, args.height, args)
    
    print_controls()
    
    def reset_simulation():
        nonlocal fluid
        fluid = FluidSimulation(args.width, args.height, args)
        center = fluid.N // 2
        radius = 15
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if i*i + j*j <= radius*radius:
                    if center + i < fluid.N and center + j < fluid.N and center + i >= 0 and center + j >= 0:
                        fluid.add_density(center + i, center + j, 100, temp=80)
                        fluid.add_velocity(center + i, center + j, 30 * (j/radius), -30 * (i/radius))
    
    reset_simulation()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    reset_simulation()
                elif event.key == pygame.K_a:
                    fluid.show_arrows = not fluid.show_arrows
                elif event.key == pygame.K_UP:
                    fluid.vorticity += 0.05
                elif event.key == pygame.K_DOWN:
                    fluid.vorticity = max(0, fluid.vorticity - 0.05)
                elif event.key == pygame.K_RIGHT:
                    fluid.buoyancy += 0.05
                elif event.key == pygame.K_LEFT:
                    fluid.buoyancy = max(0, fluid.buoyancy - 0.05)
                elif event.key == pygame.K_b:
                    fluid.cooling_rate = min(0.999, fluid.cooling_rate + 0.001)
                elif event.key == pygame.K_n:
                    fluid.cooling_rate = max(0.9, fluid.cooling_rate - 0.001)
                elif event.key == pygame.K_v:
                    fluid.velocity_damping = min(0.999, fluid.velocity_damping + 0.001)
                elif event.key == pygame.K_c:
                    fluid.velocity_damping = max(0.9, fluid.velocity_damping - 0.001)

        # Update window title with current parameters
        title = (f"Fluid Simulation - Vorticity: {fluid.vorticity:.2f}, "
                f"Buoyancy: {fluid.buoyancy:.2f}, "
                f"Cooling: {fluid.cooling_rate:.3f}, "
                f"Damping: {fluid.velocity_damping:.3f}")
        pygame.display.set_caption(title)

        fluid.step()
        screen.fill((0, 0, 0))
        fluid.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
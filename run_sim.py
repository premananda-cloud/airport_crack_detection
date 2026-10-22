import time
from src.sim.world import SimulationWorld

def main():
    print("Initializing PyBullet Simulation World...")
    world = SimulationWorld(
        runway_length=100.0,
        runway_width=6.0,
        inspection_altitude=15.0,
        gui=True
    )
    
    print("World loaded. Running simulation loop for 10 seconds...")
    start_time = time.time()
    try:
        while time.time() - start_time < 10.0:
            world.step()
            
            # Optional: Monitor drone state
            # pos, orn, vel, ang_vel = world.get_drone_state()
            # print(f"Drone Pos: {pos}")
            
            time.sleep(1.0/100.0) # Sync real-time with 100Hz sim step
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        world.disconnect()
        print("Simulation closed.")

if __name__ == "__main__":
    main()
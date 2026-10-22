import pybullet as p
import pybullet_data
import os

class SimulationWorld:
    """
    PyBullet simulation world for airport runway inspection.
    Based on PLAN.md and MATH.md specifications.
    """
    def __init__(self, runway_length=100.0, runway_width=6.0, inspection_altitude=15.0, gui=True):
        self.runway_length = runway_length
        self.runway_width = runway_width
        self.inspection_altitude = inspection_altitude
        
        # Connect to PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0/100.0) # 100 Hz as per PLAN.md
        
        self._load_environment()
        self._load_drone()
        self._setup_camera()

    def _load_environment(self):
        # 1. Ground Plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Apply asphalt texture
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        tex_path = os.path.join(project_root, 'asphalt_tex.ppm')
        if os.path.exists(tex_path):
            texture_id = p.loadTexture(tex_path)
            p.changeVisualShape(self.ground_id, -1, textureUniqueId=texture_id)
        else:
            print(f"Warning: Texture not found at {tex_path}")
            
        # 2. Runway Visual Marker (100m x 6m)
        runway_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.runway_length/2, self.runway_width/2, 0.005],
            rgbaColor=[0.25, 0.25, 0.25, 1.0], 
            specularColor=[0.1, 0.1, 0.1]
        )
        self.runway_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=runway_visual,
            basePosition=[0, 0, 0.005]
        )
        
        # 3. Runway Markings
        self._add_runway_markings()
        
        # 4. Inspection Volume (Cuboid)
        self._add_inspection_volume()

    def _add_runway_markings(self):
        # Centerline (dashed)
        dash_length = 2.0
        gap_length = 2.0
        num_dashes = int(self.runway_length / (dash_length + gap_length))
        
        for i in range(num_dashes):
            x_pos = -self.runway_length/2 + dash_length/2 + i * (dash_length + gap_length)
            dash_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[dash_length/2, 0.1, 0.001],
                rgbaColor=[1.0, 1.0, 1.0, 1.0]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=dash_visual,
                basePosition=[x_pos, 0, 0.011]
            )
            
        # Edge lines (solid)
        for y in [-self.runway_width/2 + 0.15, self.runway_width/2 - 0.15]:
            edge_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[self.runway_length/2, 0.15, 0.001],
                rgbaColor=[1.0, 1.0, 1.0, 1.0]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=edge_visual,
                basePosition=[0, y, 0.011]
            )

    def _add_inspection_volume(self):
        # Visual wireframe for the inspection volume
        # 100m x 6m x 20m (altitude 15m + 5m margin as per MATH.md)
        vol_height = 20.0
        vol_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.runway_length/2, self.runway_width/2, vol_height/2],
            rgbaColor=[0.0, 1.0, 0.0, 0.05], # Semi-transparent green
            specularColor=[0.0, 0.0, 0.0]
        )
        self.volume_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vol_visual,
            basePosition=[0, 0, vol_height/2]
        )

    def _load_drone(self):
        # Create a simple box to represent the drone for stage one
        # Dimensions: 0.4m x 0.4m x 0.1m
        col_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05])
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX, 
            halfExtents=[0.2, 0.2, 0.05], 
            rgbaColor=[0.2, 0.6, 0.8, 1.0]
        )
        
        self.drone_id = p.createMultiBody(
            baseMass=1.5, # Approx 1.5kg
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, self.inspection_altitude],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )

    def _setup_camera(self):
        # Position debug camera to look at the runway
        p.resetDebugVisualizerCamera(
            cameraDistance=25.0,
            cameraYaw=0.0,
            cameraPitch=-45.0,
            cameraTargetPosition=[0, 0, 5.0]
        )

    def step(self):
        p.stepSimulation()

    def get_drone_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)
        return pos, orn, vel, ang_vel

    def disconnect(self):
        p.disconnect(self.physics_client)
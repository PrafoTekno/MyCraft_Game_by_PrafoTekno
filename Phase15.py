
# The best version overall
# Cmaera control is great
# Can combine, change object, and there's gravity
# Now the game has 6 kind of object. 
# Judul bisa Pengembangan Game dengan Hand Detection untuk melatih motorik anak

import cv2
import math
from random import *
from cvzone.HandTrackingModule import HandDetector
from ursina import *

# -------------------------
# CVZone setup
# -------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7, maxHands=2)
thumb_up  = False       # left hand (reverse for right hand)
index_up  = False 
middle_up = False 
ring_up   = False 
pinky_up  = False 

# -------------------------
# Ursina setup
# -------------------------
app = Ursina()
window.color = color.rgb(30/255, 30/255, 30/255)
#window.show_fps = True
fps_text = Text(position=(-0.85, 0.45), scale=1)

# --- MAIN LIGHT ---
light = DirectionalLight(shadows=True)
light.rotation = Vec3(55, -45, 0)      # angled light (best detail visibility)
light.shadow_strength = 0.9            # softer shadows
light.shadow_resolution = 4096          # ultra clean shadows

# --- AMBIENT FILL LIGHT ---
AmbientLight(color=color.rgba(150/255, 150/255, 150/255, 1))

ground = Entity(
    position = Vec3 (0,0,0),
    model='plane',
    scale=20,
    texture='white_cube',
    colider = 'box',
    color=color.rgb(160/255, 160/255, 160/255)
)

#----------------
# Camera setting
#----------------

# For zoom out/in
zoom_speed = 0
zoom_amount = 25
min_zoom = 0
max_zoom = 50
pinch_reference = None
zoom_active = False
forward_dir = 0

# Create pivot for stable orbit
pivot = Entity(position=(0,0,0))
pivot.rotation = (16,0,0)
orbit_pitch = 20      # starting angle
orbit_yaw = 0

camera.parent = pivot
camera.position = (0, 8, -25)
camera.look_at(ground)

print (f"Camera rot = {camera.position}")
print (f"light rot = {light.position}")

# For rotation and pan
friction = 0.5
sensitivity = 2.5
threshold = 5
movement_threshold = 7

rotation_speed_x = 0
rotation_speed_y = 0

reference_x = None
reference_y = None
hand_active = False

# ----- Pan -------
previous_mode = ""
ref_pan_x = None
ref_pan_y = None
position_speed_x = 0
position_speed_z = 0
hand_pan_active = False

pan_velocity = Vec3(0,0,0)
target_move = Vec3(0,0,0)

def smooth(old, new, factor=0.18):
    return old + (new - old) * factor

# -------------------------
# Hand Visualization (UI)
# -------------------------
class HandVisualization:
    def __init__(self):
        self.lms = [Entity(parent=camera.ui, model='sphere', color=color.azure, scale=0.03, enabled=False) for _ in range(21)]

        self.CONN = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]
        self.bones = [Entity(parent=camera.ui, model='cube', color=color.orange, scale=(0.02,0.02,1), enabled=False)
                      for _ in self.CONN]

    def update(self, lm, cam_to_ui_fn):
        if lm is None:
            self.hide()
            return

        pts = [cam_to_ui_fn(x,y,z) for (x,y,z) in lm]

        for i,p in enumerate(pts):
            self.lms[i].position = p
            self.lms[i].enabled = True

        for i,(a,b) in enumerate(self.CONN):
            p1 = pts[a]
            p2 = pts[b]
            bone = self.bones[i]
            mid = (p1+p2)/2
            bone.position = mid
            bone.look_at(p2)
            bone.scale_z = max(0.001, distance(p1,p2))
            bone.enabled = True

    def hide(self):
        for s in self.lms: s.enabled=False
        for b in self.bones: b.enabled=False

hand_vis = HandVisualization()

# -------------------------
# Change Color
# -------------------------

AREA_CENTER_COLOR_1 = Vec3(6, 0.1, -3)
AREA_CENTER_COLOR_2 = Vec3(6, 0.1, -6)
AREA_CENTER_COLOR_RESULT = Vec3(6, 0.1, 0)
AREA_RADIUS = 1

def in_color_area (box):    
    if AREA_CENTER_COLOR_RESULT.x - AREA_RADIUS <= box.x <= AREA_CENTER_COLOR_RESULT.x + AREA_RADIUS and AREA_CENTER_COLOR_RESULT.z - AREA_RADIUS <= box.z <= AREA_CENTER_COLOR_RESULT.z + AREA_RADIUS:
        return 1
    else:
        return 0

color_1 = {
    1: color.white,
    2: color.red,
    3: color.yellow,
    4: color.blue
}
color_2 = {
    1: color.white,
    2: color.red,
    3: color.yellow,
    4: color.blue
}
color_result = {
    1: color.white,
    2: color.red,
    3: color.yellow,
    4: color.blue,
    6: color.orange,
    8: color.rgba (252/255, 7/255, 229/255, 0.8),
    12: color.green
}

c1 = 1
c2 = 2
cR = 2

# ----- Set Place -----
set_Result_COLOR = Entity (
    model='plane',
    color=color_result[cR],
    position=AREA_CENTER_COLOR_RESULT,
    scale=(AREA_RADIUS*2, 0.1, AREA_RADIUS*2),
    collider=None
)

set_COLOR_1 = Entity (
    model='cube',
    color=color_1[c1],
    position=AREA_CENTER_COLOR_1,
    scale=(AREA_RADIUS*2, 0.1, AREA_RADIUS*2),
    collider='box',
)
set_COLOR_1.is_button=True
set_COLOR_1.which_color_button=1

set_COLOR_2 = Entity (
    model = "cube",
    color=color_2[c2],
    position=AREA_CENTER_COLOR_2,
    scale=(AREA_RADIUS*2, 0.1, AREA_RADIUS*2),
    collider='box',
)
set_COLOR_2.is_button=True
set_COLOR_2.which_color_button=2

# -------------------------
# Pointer cursor + highlight
# -------------------------
pointer = Entity(
    model='sphere',
    scale=0.07,
    color=color.azure,
    position=(0, 0, 0),
    enabled=False
)

highlight_box = None

def highlight_entity(entity):
    global highlight_box

    # Clear old highlight
    if highlight_box and highlight_box != entity:
        highlight_box.color = highlight_box.original_color

    # If new entity to highlight
    if entity:
        if not hasattr(entity, "original_color"):
            entity.original_color = entity.color
        entity.color = color.yellow
        highlight_box = entity
    else:
        highlight_box = None

# -------------------------
# Box management
# -------------------------
spawned_object = []
selected_box = None
is_grabbing = False

Spawn_Area = Entity (
    position = Vec3 (6, 0.1, 5),
    model = "plane",
    scale=5,
    texture='white_cube',
    colider = None,
    color=color.rgb(0, 200, 0)
)

def spawn_ball(n):
    for _ in range(n):
        b = Entity(
            model='Assets/MyCraft_Game/1.glb',
            color=color_result[cR],
            scale=0.5,
            position=(uniform(Spawn_Area.x-(5/2), Spawn_Area.x+(5/2)), 0.4, uniform(Spawn_Area.z-(5/2), Spawn_Area.z+(5/2))),
            collider='sphere',
            gravity = True
        )
        b.velocity_y = 0         # add vertical velocity
        b.current_color = color_result[2]
        b.color_ID = cR
        b.type = 1
        spawned_object.append(b)

# -------------------------
# Combine Object
# -------------------------

count_obj = 0
ref_obj_x = 0
ref_obj_z = 0
obj_in_combine_Area = False

def Is_in_Combine_Area(obj):
    
    global ref_obj_x, ref_obj_z, obj_in_combine_Area

    # --- SLOT B at (-7, -6) ---
    if (-7-AREA_RADIUS*2) <= obj.x <= (-7+AREA_RADIUS*2) and (-6-AREA_RADIUS*2) <= obj.z <= (-6+AREA_RADIUS*2):
        obj.x = -7
        obj.z = -6
        obj_in_combine_Area = True

    # --- SLOT A at (-7, -3) ---
    elif (-7-AREA_RADIUS*2) <= obj.x <= (-7+AREA_RADIUS*2) and (-3-AREA_RADIUS*2) <= obj.z <= (-3+AREA_RADIUS*2):
        obj.x = -7
        obj.z = -3
        obj_in_combine_Area = True

    else:
        obj_in_combine_Area = False

    # --- If object is inside slot ---
    if obj_in_combine_Area and (ref_obj_x != obj.x or ref_obj_z != obj.z):

        # mark slot as filled
        ref_obj_x = obj.x
        ref_obj_z = obj.z

        # return object type
        if obj.type == 1:  return 1
        if obj.type == 2:  return 2
        if obj.type == 3:  return 3
        if obj.type == 4:  return 4
        if obj.type == 5:  return 5
        if obj.type == 6:  return 6
        return None

    # --- If object LEFT the slot (only reset if this object was stored) ---
    else:
        if ref_obj_x == obj.x and ref_obj_z == obj.z:
            ref_obj_x = 0
            ref_obj_z = 0

        return None

def Combine_Active (kind_1, kind_2, n):
    if n == 2:
        if kind_1 + kind_2 <= 6:
            return kind_1 + kind_2
        else:
            return None

def Combine_Result (obj, kind):

    scl = obj.scale
    vel = getattr(obj, 'velocity_y', 0)

    if kind == 1:
        # create new Ball
        Ball = Entity( #235, 52, 125
            model='Assets/MyCraft_Game/1.glb',
            color=color.rgb(235/255, 52/255, 125/255),
            position=Vec3 (-7, 0.1, 0),
            scale=scl,
            collider='sphere'
        )
        Ball.type = 1
        Ball.current_color = color.rgb(235/255, 52/255, 125/255)
        Ball.velocity_y = vel
        Ball.color_ID = 0
        Ball.gravity = True
        spawned_object.append(Ball)

    elif kind == 2:
        # create new Silinder
        Silinder = Entity( #235, 52, 125
            model='Assets/MyCraft_Game/2.glb',
            color=color.rgb(235/255, 52/255, 125/255),
            position=Vec3 (-7, 0.1, 0),
            scale=scl,
            collider='box'
        )
        Silinder.type = 2
        Silinder.current_color = color.rgb(235/255, 52/255, 125/255)
        Silinder.velocity_y = vel
        Silinder.color_ID = 0
        Silinder.gravity = True
        spawned_object.append(Silinder)

    elif kind == 3:
        # create new Triangle
        Triangle = Entity(
            model='Assets/MyCraft_Game/3.glb',
            color=color.rgb(247/255, 144/255, 144/255),
            position=Vec3 (-7, 0.1, 0),
            scale=scl*1.5,
            collider='mesh'
        )
        Triangle.type = 3
        Triangle.current_color = color.rgb(247/255, 144/255, 144/255)
        Triangle.velocity_y = vel
        Triangle.color_ID = 0
        Triangle.gravity = True
        spawned_object.append(Triangle)

    elif kind == 4:
        # create new Box
        Box = Entity( #27, 247, 130
            model='Assets/MyCraft_Game/4.glb',
            color=color.rgb(27/255, 247/255, 130/255),
            position=Vec3 (-7, 0.1, 0),
            scale=scl,
            collider='mesh'
        )
        Box.type = 4
        Box.current_color = color.rgb(27/255, 247/255, 130/255)
        Box.velocity_y = vel
        Box.color_ID = 0
        Box.gravity = True
        spawned_object.append(Box)

    elif kind == 5:
        # create new Pentagon
        Pentagon = Entity( #27, 247, 130
            model='Assets/MyCraft_Game/5.glb',
            color=color.rgb(27/255, 247/255, 130/255),
            position=Vec3 (-7, 0.1, 0),
            scale=scl,
            collider='mesh'
        )
        Pentagon.type = 5
        Pentagon.current_color = color.rgb(27/255, 247/255, 130/255)
        Pentagon.velocity_y = vel
        Pentagon.color_ID = 0
        Pentagon.gravity = True
        spawned_object.append(Pentagon)

    elif kind == 6:
        # create new Hexagon
        Hexagon = Entity( #27, 247, 130
            model='Assets/MyCraft_Game/6.glb',
            color=color.rgb(27/255, 247/255, 130/255),
            position=Vec3 (-7, 0.1, 0),
            scale=scl,
            collider='mesh'
        )
        Hexagon.type = 6
        Hexagon.current_color = color.rgb(27/255, 247/255, 130/255)
        Hexagon.velocity_y = vel
        Hexagon.color_ID = 0
        Hexagon.gravity = True
        spawned_object.append(Hexagon)

Slot_A = Entity (
    position = Vec3 (-7, 0.1, -3),
    model = "cube",
    scale=(2,0.1,2),
    texture='white_cube',
    colider = None,
    color=color.rgb(241/255, 177/255, 66/255)
)

Slot_B = Entity (
    position = Vec3 (-7, 0.1, -6),
    model = "cube",
    scale=(2,0.1,2),
    texture='white_cube',
    colider = None,
    color=color.rgb(241/255, 177/255, 66/255)
)

Slot_Result = Entity (
    position = Vec3 (-7, 0.1, 0),
    model = "cube",
    scale=(2,0.1,2),
    texture='white_cube',
    colider = None,
    color=color.rgb(241/255, 177/255, 66/255)
)

# -------------------------
# Gravity Logic
# -------------------------

def apply_gravity(obj):
    if not hasattr(obj, "gravity") or not obj.gravity:
        return

    # Gravity logic
    obj.velocity_y -= 3.5 * time.dt
    obj.y += obj.velocity_y * time.dt

    # Collision with ground (y=0)
    if obj.y <= 0.4:
        obj.y = 0.4
        obj.velocity_y = 0

# -------------------------
# 2D UI mapping + projection
# -------------------------
def cam_to_ui(px, py, img_w, img_h):
    nx = px/img_w
    ny = py/img_h
    return Vec3( (nx-0.5)*0.9, (0.5-ny)*0.9, 0 )

def ui_to_world(x_ui, y_ui, dist=2):
    cam = camera
    return (
        cam.world_position
        + cam.forward * dist
        + cam.right * (x_ui * dist * 1.3)
        + cam.up * (y_ui * dist * 1.3)
    )

debug_text = Text("", position=window.top_left + (0.02,-0.02), scale=0.9)

last_right_count = 0
count_fing_up = 0

# -------------------------
# What's the command?
# -------------------------

AREA_CENTER_FIRST_COMMAND = Vec3(-5, 0.1, 5)
AREA_CENTER_SECOND_COMMAND = Vec3(-2, 0.1, 5)
AREA_CENTER_THIRD_COMMAND = Vec3(1, 0.1, 5)

# Generate random command for each slot

def generate_task (shape, color, x,y,z):

    if shape == 1:
        # create new Ball
        Ball = Entity( #235, 52, 125
            model='Assets/MyCraft_Game/1.glb',
            color=color_result[color],
            position=Vec3(x,y,z),
            rotation=Vec3(-90,0,0),
            scale=0.4,
            collider='sphere'
        )
        return Ball


    elif shape == 2:
        # create new Silinder
        Silinder = Entity( #235, 52, 125
            model='Assets/MyCraft_Game/2.glb',
            color=color_result[color],
            position=Vec3(x,y,z),
            rotation=Vec3(-90,0,0),
            scale=0.4,
            collider='box'
        )
        return Silinder

    elif shape == 3:
        # create new Triangle
        Triangle = Entity(
            model='Assets/MyCraft_Game/3.glb',
            color=color_result[color],
            position=Vec3(x,y,z),
            rotation=Vec3(-90,0,0),
            scale=0.4,
            collider='mesh'
        )
        return Triangle

    elif shape == 4:
        # create new Box
        Box = Entity( #27, 247, 130
            model='Assets/MyCraft_Game/4.glb',
            color=color_result[color],
            position=Vec3(x,y,z),
            rotation=Vec3(-90,0,0),
            scale=0.4,
            collider='mesh'
        )
        return Box

    elif shape == 5:
        # create new Pentagon
        Pentagon = Entity( #27, 247, 130
            model='Assets/MyCraft_Game/5.glb',
            color=color_result[color],
            position=Vec3(x,y,z),
            rotation=Vec3(-90,0,0),
            scale=0.4,
            collider='mesh'
        )
        return Pentagon

    elif shape == 6:
        # create new Hexagon
        Hexagon = Entity( #27, 247, 130
            model='Assets/MyCraft_Game/6.glb',
            color=color_result[color],
            position=Vec3(x,y,z),
            rotation=Vec3(-90,0,0),
            scale=0.4,
            collider='mesh'
        )
        return Hexagon

command = [None] * 3
random_shape = [None] * 3
random_color = [None] * 3

location_task = [None] * 3
location_task[0] = AREA_CENTER_FIRST_COMMAND
location_task[1] = AREA_CENTER_SECOND_COMMAND
location_task[2] = AREA_CENTER_THIRD_COMMAND

def generate_commands():
    for i in range(3):

        if command[i] is not None:
            destroy(command[i])

        random_shape[i] = randint(1, 6)
        random_color[i] = choice([1, 2, 3, 4, 6, 8, 12])

        area = location_task[i]
        command[i] = generate_task(
            random_shape[i],
            random_color[i],
            area.x,
            area.y + 2,
            area.z
        )

generate_commands ()

def check_task(obj):

    # Scan each task slot
    for i in range(len(location_task)):
        # Check if object is inside slot area
        if (location_task[i].x-AREA_RADIUS*2) <= obj.x <= (location_task[i].x+AREA_RADIUS*2) and \
           (location_task[i].z-AREA_RADIUS*2) <= obj.z <= (location_task[i].z+AREA_RADIUS*2):

            # Snap object to center
            obj.x = location_task[i].x
            obj.z = location_task[i].z

            # Correct shape AND correct color
            if obj.type == random_shape[i] and obj.color_ID == random_color[i]:
                return 1      # correct
            else:
                return 0      # inside slot, but wrong object

    return None  # outside all slots
    
def gift ():

    Gift = Entity (
            model='Assets/MyCraft_Game/Monkey.glb',
            position=Vec3 (0,0.3,0),
            texture = 'white_cube',
            color=color.rgb (230/255, 200/255, 41/255),
            rotation=Vec3 (0,0,0),
            scale=1.2,
            collider='box'
    )
    Gift.type = 10
    Gift.velocity_y = 0
    Gift.gravity = True
    spawned_object.append(Gift)

First = Entity (
    position = AREA_CENTER_FIRST_COMMAND,
    model = "plane",
    scale=AREA_RADIUS*2,
    texture='white_cube',
    colider = None,
    color=color.rgb(157/255, 0/255, 166/255)
)

Second = Entity (
    position = AREA_CENTER_SECOND_COMMAND,
    model = "plane",
    scale=AREA_RADIUS*2,
    texture='white_cube',
    colider = None,
    color=color.rgb(157/255, 0/255, 166/255)
)

Third = Entity (
    position = AREA_CENTER_THIRD_COMMAND,
    model = "plane",
    scale=AREA_RADIUS*2,
    texture='white_cube',
    colider = None,
    color=color.rgb(157/255, 0/255, 166/255)
)

# -------------------------
# Update loop
# -------------------------
def update():

    fps_text.text = f"FPS: {int(1 / time.dt)}"

    global selected_box, is_grabbing, last_right_count, count_fing_up
    global rotation_speed_x, rotation_speed_y, reference_x, reference_y, hand_active
    global ref_pan_x, ref_pan_y, hand_pan_active, previous_mode, position_speed_x, position_speed_z
    global zoom_speed, max_zoom, min_zoom, zoom_amount, pinch_reference, zoom_active
    global thumb_up, index_up, middle_up, ring_up, pinky_up
    global highlight_box, pan_velocity, target_move, forward_dir
    global c1, c2, cR

    hover_button = None

    success, img = cap.read()
    if not success:
        return

    img = cv2.flip(img, 1)
    h, w = img.shape[:2]

    # Detect hands (flipType=False because we already flipped manually)
    hands, img = detector.findHands(img, draw=True, flipType=False)

    # Fix handedness due to manual flip
    def fix_type(hnd):
        return "Right" if hnd["type"] == "Left" else "Left"

    left = None
    right = None

    if hands:
        for hnd in hands:
            t = fix_type(hnd)
            if t == "Left":
                left = hnd
            else:
                right = hnd


    # ------------- WHICH FINGER UP LEFT -----------------------
    current_mode_R = "none"

    if right and right.get("lmList") and len(right["lmList"]) >= 21:

        thumb_up  = right["lmList"][4][0] > right["lmList"][3][0]     # right hand
        index_up  = right["lmList"][8][1] < right["lmList"][6][1]
        middle_up = right["lmList"][12][1] < right["lmList"][10][1]
        ring_up   = right["lmList"][16][1] < right["lmList"][14][1]
        pinky_up  = right["lmList"][20][1] < right["lmList"][18][1]

        if not thumb_up and not index_up:
            current_mode_R = "spawn_box"
        elif not middle_up and not ring_up and not pinky_up:
            current_mode_R = "selected_box"
        elif not middle_up and not ring_up and not pinky_up and thumb_up and not index_up:
            current_mode_R = "switch_obj"
        else:
            current_mode_R = "none"

    # ---------------- LEFT HAND: SPAWN BOXES ----------------
    right_count = 0
    if current_mode_R == "spawn_box":
        right_count = sum(detector.fingersUp(right))
        if 1 <= right_count <= 3 and right_count != last_right_count:
            spawn_ball(right_count-1)

    last_right_count = right_count

    # ---------------- LEFT HAND: POINTER / RAYCAST / GRAB ----------------
    if current_mode_R == "selected_box":

        lm = right["lmList"]
        thumb = lm[4]
        index = lm[8]
        wrist = lm[0]
        ibase = lm[5]

        pinch_px = math.dist(thumb, index)
        pinch_thresh = 70

        # ---- Convert finger tip to world space ----
        ui_pt = cam_to_ui(index[0], index[1], w, h)
        world_pt = ui_to_world(ui_pt.x, ui_pt.y, dist=3.0)

        # ---- Update pointer cursor ----
        pointer.enabled = True
        pointer.position = world_pt
        
        # ---- RAYCAST ----
        ray_dir = (world_pt - camera.world_position).normalized()
        hit = raycast(
            camera.world_position,
            ray_dir,
            distance=15,
            ignore=[camera],
            debug=False
        )

        hover_box = None

        if hit.hit:
            pointer.position = hit.world_point
        else:
            pointer.position = camera.world_position + ray_dir * 6
        
        hover_box = hit.entity if (hit.hit and hit.entity in spawned_object) else None
        # ---- Highlight hovered box ----
        highlight_entity(hover_box)

        hover_button = hit.entity if (hit.hit and getattr(hit.entity, 'is_button', False)) else None

        #print (f"hover_box = {hover_box}")
        #print (f"hover_button = {hover_button}")

        # ---------------- PINCH GRAB ----------------
        if pinch_px < pinch_thresh:

            # Start grabbing Object or click button
            if not is_grabbing:
                selected_box = None

                # Use raycast result
                if hover_box:
                    selected_box = hover_box
                    is_grabbing = True

            # Move grabbed box
            else:
                if selected_box:

                    # Smooth move
                    selected_box.position = lerp(
                        selected_box.position, 
                        world_pt, 
                        time.dt * 10
                    )

                    # Rotate box (hand twist)
                    dx = ibase[0] - wrist[0]
                    dz = ibase[1] - wrist[1]
                    angle = math.degrees(math.atan2(dx, dz))
                    selected_box.rotation_y = lerp(
                        selected_box.rotation_y, 
                        angle, 
                        time.dt * 10
                    )

        else:
            # Release
            is_grabbing = False
            selected_box = None

    else:
        is_grabbing = False
        selected_box = None
        pointer.enabled = False

    n_obj = []
    obj_true = []

    if not is_grabbing:

        # ------ Apply gravity to all object ----- #
        for e in scene.entities:
            if hasattr(e, "gravity") and e.gravity:
                apply_gravity(e)

        # -------- Change Color ------- #
        if hover_button and pinch_px < (pinch_thresh + 15):

            changed = False

            if hover_button.which_color_button == 1:
                c1 = c1 % 4 + 1
                hover_button.color = color_1[c1]
                changed = True

            elif hover_button.which_color_button == 2:
                c2 = c2 % 4 + 1
                hover_button.color = color_2[c2]
                changed = True

            if changed:
                if c1 == c2:
                    cR = c1
                else:
                    cR = c1 * c2
                set_Result_COLOR.color = color_result[cR]

        for e in spawned_object:
            if in_color_area(e) == 1:
                e.color = color_result[cR]
                e.current_color = color_result[cR]
                e.color_ID = cR
            else:
                if hasattr(e, 'current_color'):
                    e.color = e.current_color

        # ----- Combine Object ---- #
        to_remove = []

        for e in spawned_object:
            t = Is_in_Combine_Area(e)

            if t is not None:
                n_obj.append((e, t))

                if len(n_obj) == 2:
                    obj1, kind1 = n_obj[0]
                    obj2, kind2 = n_obj[1]

                    Kind = Combine_Active(kind1, kind2, 2)

                    if Kind is not None:
                        Combine_Result(obj1, Kind)
                        to_remove += [obj1, obj2]

                    n_obj.clear()

        # Remove after loop (AMAN)
        for obj in to_remove:
            if obj in spawned_object:
                spawned_object.remove(obj)
            destroy(obj)

        # ------- Command Tasks ------- #
        for e in spawned_object:
            result = check_task(e)

            if result == 1:
                if e not in obj_true:
                    obj_true.append(e)

        # If player placed 3 correct objects
        if len(obj_true) == 3:
            gift ()
            
            # ---- REMOVE the 3 correct objects ----
            for correct_obj in obj_true:
                if correct_obj in spawned_object:
                    spawned_object.remove(correct_obj)
                destroy(correct_obj)

            # Reset
            obj_true.clear()
            generate_commands ()


    # ------------- WHICH FINGER UP RIGHT -----------------------
    current_mode_L = "none"

    if left and left.get("lmList") and len(left["lmList"]) >= 21:

        thumb_up  = left["lmList"][4][0] < left["lmList"][3][0]   
        index_up  = left["lmList"][8][1] < left["lmList"][6][1]
        middle_up = left["lmList"][12][1] < left["lmList"][10][1]
        ring_up   = left["lmList"][16][1] < left["lmList"][14][1]
        pinky_up  = left["lmList"][20][1] < left["lmList"][18][1]

        # ---- Mode gestures ----
        if index_up and thumb_up and not middle_up and not ring_up and not pinky_up:
            current_mode_L = "zoom"
        elif index_up and not middle_up and not thumb_up and not ring_up and not pinky_up:
            current_mode_L = "rotate"
        elif index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
            current_mode_L = "pan"
        else:
            current_mode_L = "none"


    # -------- Reset camera when pinky up is closed ----------
    if left and left.get("lmList") and pinky_up and not thumb_up and not index_up and not middle_up and not ring_up:
        camera.position = (0, 8, -25)
        pivot.position = (0, 0, 0)
        pivot.rotation = (16, 0, 0)

        reference_x = None
        reference_y = None
        hand_active = False
        rotation_speed_x = 0
        rotation_speed_y = 0
        pinch_reference = None

        ref_pan_x = None
        ref_pan_y = None
        position_speed_x = 0
        position_speed_z = 0
        hand_pan_active = False
        pan_velocity = (0,0,0)

    
    # =====================================================
    # MODE CHANGE → RESET ONLY THE VARIABLES OF THAT MODE
    # =====================================================
    if current_mode_L != previous_mode:
    
        # Rotation references
        if current_mode_L == "rotate":
            reference_x = None
            reference_y = None
            hand_active = False
            rotation_speed_x = 0
            rotation_speed_y = 0

        # Zoom references
        if current_mode_L == "zoom":
            pinch_reference = None

        # Pan references
        # (Important: pan uses SEPARATE refs!)
        if current_mode_L == "pan":
            ref_pan_x = None
            ref_pan_y = None
            position_speed_x = 0
            position_speed_z = 0
            hand_pan_active = False

        previous_mode = current_mode_L

    # ==================================================================
    # 1. ROTATION MODE --------------------------------------------------
    # ==================================================================
    if current_mode_L == "rotate":

        ix, iy = left["lmList"][8][:2]

        if reference_x is None:
            reference_x = ix
            reference_y = iy
            hand_active = False
            return

        dx = ix - reference_x
        dy = iy - reference_y

        if not hand_active:
            if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
                hand_active = True
            else:
                reference_x = ix
                reference_y = iy
                return

        # Dominant axis filtering
        target_rot_x = 0
        target_rot_y = 0

        if abs(dx) > abs(dy) and abs(dx) > threshold:
            target_rot_y = dx * sensitivity
        elif abs(dy) > threshold:
            target_rot_x = dy * (sensitivity-0.6)

        # Smooth rotation
        rotation_speed_x = smooth(rotation_speed_x, target_rot_x)
        rotation_speed_y = smooth(rotation_speed_y, target_rot_y)

        pivot.rotation_y += rotation_speed_y
        pivot.rotation_x = clamp(pivot.rotation_x + rotation_speed_x, 0.2, 80)

        reference_x = ix
        reference_y = iy

    light.rotation_x = 55                       # stays angled downward
    light.rotation_y = camera.rotation_y - 60   # offset for realism

    # ==================================================================
    # 2. ZOOM MODE ------------------------------------------------------
    # ==================================================================
    if current_mode_L == "zoom":

        index = left["lmList"][8]
        thumb = left["lmList"][4]

        dx = index[0] - thumb[0]
        dy = index[1] - thumb[1]
        pinch = math.hypot(dx, dy)

        if pinch_reference is None:
            pinch_reference = pinch
            return

        d_pinch = pinch - pinch_reference
        pinch_reference = pinch

        zoom_step = d_pinch * 0.57

        forward_dir = camera.forward.normalized()
        proposed = camera.world_position + forward_dir * zoom_step

        hit = raycast(camera.world_position, forward_dir,
                  distance=abs(zoom_step) + 0.5, ignore=[camera])

        if hit.hit and zoom_step > 0:
            if distance(camera.world_position, hit.world_point) < 1.0:
                proposed = camera.world_position

        # smooth zoom
        camera.world_position = smooth(camera.world_position, proposed, factor=0.22)

    # ==================================================================
    # 3. PAN MODE ------------------------------------------------------
    # ==================================================================
    if current_mode_L == "pan":

        ty, tx = left["lmList"][12][:2]

        if ref_pan_x is None:
            ref_pan_x = tx
            ref_pan_y = ty
            hand_pan_active = False
            position_speed_x = 0
            position_speed_z = 0
            return

        dx = tx - ref_pan_x
        dz = ty - ref_pan_y

        if not hand_pan_active:
            if abs(dx) > movement_threshold or abs(dz) > movement_threshold:
                hand_pan_active = True
            else:
                ref_pan_x = tx
                ref_pan_y = ty
                return

        dist = distance(camera.world_position, pivot.world_position)
        pan_scale = dist * 0.007

        # Dominant axis filter
        if abs(dx) > abs(dz) and abs(dx) > threshold:
            position_speed_z = -dx * pan_scale
            position_speed_x = 0
        elif abs(dz) > threshold:
            position_speed_x = dz * pan_scale
            position_speed_z = 0
        else:
            position_speed_x = 0
            position_speed_z = 0

        # --- APPLY PAN TO PIVOT ---
        pivot.x += position_speed_x
        pivot.z += position_speed_z

        if  pivot.x <= -13:
            pivot.x = -13
        if pivot.x >= 13:
            pivot.x = 13
        if  pivot.z <= -13:
            pivot.z = -13
        if pivot.z >= 13:
            pivot.z = 13

        ref_pan_x = tx
        ref_pan_y = ty

        #print (f"pan_scale = {pan_scale:.2f} | x = {pivot.x:.2f} | z = {pivot.z:.2f}")
        # ---- SET TARGET SMOOTH MOVEMENT ----
        target_move = Vec3(position_speed_x, 0, position_speed_z)
        # interpolate velocity toward target
        pan_velocity = lerp(pan_velocity, target_move, 4 * time.dt)  
        # (4 * dt gives ~0.25 smoothing but framerate-independent)
        # apply velocity
        pivot.world_position += pan_velocity
    
    # ============================================================
    # ================= Hand Visualization Update ================
    # ============================================================
    if hands:
        lm = hands[0]["lmList"]
        norm = [(lm[i][0], lm[i][1], 0) for i in range(21)]
        hand_vis.update(norm, lambda x,y,z: cam_to_ui(x,y,w,h))
    else:
        hand_vis.hide()

    cv2.imshow("cvzone", img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        application.quit()

def input(key):
    if key=='escape':
        application.quit()

app.run()
cv2.destroyAllWindows()
cap.release()

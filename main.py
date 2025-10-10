import tkinter as tk
from PIL import Image, ImageTk
import heapq
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from config import HOLE_COST, PUDDLE_COST, WALLS, SCIANA, OBSTACLES, CHARGING_STATION, ROOMS
from digit_classifier import DigitClassifier
import random
import os
from genetic_alg import best_route
import numpy as np


class GridApp:
    def __init__(self, root, rows=15, cols=11, cell_size=50):
        self.root = root
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.packages = []
        # Lista pól, gdzie znajdują się regały (ściany)
        self.shelf_positions = WALLS
        self.sciany = SCIANA
        self.rooms = ROOMS

        self.canvas = tk.Canvas(root, width=(cols + 1) * cell_size, height=(rows + 1) * cell_size)
        self.canvas.pack()

        self.battery_label = tk.Label(root, text="Battery: 100%", font=("Arial", 14))
        self.battery_label.pack()

        self.current_digit_label = tk.Label(root, text="Current Package: None", font=("Arial", 14))
        self.current_digit_label.pack()

        self.current_digit_image_label = tk.Label(root)
        self.current_digit_image_label.pack()

        self.charging_station = CHARGING_STATION

        self.obstacles = OBSTACLES

        self.classifier = DigitClassifier()

        self.draw_grid()
        self.shelves = {}
        self.create_shelves()
        self.draw_charging_station()

        self.current_target = None

        # wczytanie obrazków
        self.puddle_image = Image.open("puddle.png") \
            .resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.hole_image = Image.open("dziura.png") \
            .resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.puddle_tk_image = ImageTk.PhotoImage(self.puddle_image)
        self.hole_tk_image = ImageTk.PhotoImage(self.hole_image)
        # upewniamy się, że referencje do obrazków nie zostaną usunięte
        self.canvas.image_refs.append(self.puddle_tk_image)
        self.canvas.image_refs.append(self.hole_tk_image)

        self.draw_obstacles()

        # Tworzymy paczki – bez rozmiaru/masy
        for _ in range(70):
            self.create_package(position=(0, 5))

        self.forklift = Forklift(self)
        self.draw_forklift()

        # Używamy event.char dla liter (p, d) – dla strzałek korzystamy z keysym
        self.root.bind("<KeyPress>", self.forklift.user_move)
        self.update_battery_label()

        # trenujemy model
        self.model_setup()

        self.train_room_classifier()

        # testujemy move_to_target dla przykładowego celu
        self.draw_room_labels()

        self.package_loop()

    def update_current_package_image(self, letter, image_path=None, assigned_room=None):
        image = Image.open(image_path).resize((50, 50), Image.LANCZOS)
        self.current_digit_image = ImageTk.PhotoImage(image)
        self.current_digit_image_label.config(image=self.current_digit_image)
        self.current_digit_image_label.image = self.current_digit_image

        number = ord(letter) - ord('a')
        self.current_digit_label.config(text=f"Recognized as: {number}, Assigned room: {assigned_room.upper()}")

    def package_loop(self):
        while self.packages:  # Wykonuj pętlę, dopóki są paczki do obsłużenia
            self.forklift.holding_packages = []  # Resetuje wózek przed nowym cyklem

            self.move_to_target((1, 5))
            for _ in range(6):
                if not self.packages:
                    break

                if self.forklift.position == (1, 5):
                    if self.forklift.direction == 0:
                        self.forklift.rotate_left()
                    elif self.forklift.direction == 180:
                        self.forklift.rotate_right()

                self.forklift.pickup()
            if self.forklift.holding_packages:
                assigned_packages = []
                for package in self.forklift.holding_packages:
                    try:
                        target, room = self.assign_room_position(self.forklift.position, package.goods_type)
                        package.assigned_room = room
                        package.delivery_target = target
                        assigned_packages.append(package)
                    except Exception as e:
                        print(f"Pokój pełny, pomijam")

                destinations = [p.delivery_target for p in assigned_packages]
                if len(destinations) > 1:
                    optimal_order = best_route(destinations)

                    for index in optimal_order:
                        if index < len(assigned_packages):
                            package = assigned_packages[index]
                            target = package.delivery_target
                            digit = package.goods_type[-1]
                            self.update_current_package_image(digit, package.image_path, package.assigned_room)

                            self.move_to_shelf(target)
                            self.forklift.drop(package)
                else:
                    for package in assigned_packages:
                        target = package.delivery_target
                        print(f"Transport package type '{package.goods_type}' to {target}")
                        self.move_to_shelf(target)
                        self.forklift.drop(package)

    def move_to_shelf(self, shelf_pos):
        x, y = shelf_pos
        possible_positions = [(x - 1, y), (x + 1, y)]

        for index, position in enumerate(possible_positions):
            x, y = position
            if 0 <= x < self.rows and 0 <= y < self.cols:
                if position not in self.shelves and position not in self.sciany:
                    self.move_to_target(position)

                    if index == 0:
                        desired_dir = 270
                    else:
                        desired_dir = 90

                    while self.forklift.direction != desired_dir:
                        current = self.forklift.direction
                        diff = desired_dir - current

                        if diff == 0:
                            break
                        elif diff == 90:
                            self.forklift.rotate_left()
                        elif diff == 270:
                            self.forklift.rotate_right()
                        else:
                            self.forklift.rotate_right()

                    return True

        return False

    def draw_room_labels(self):
        for room_name, ((x_min, x_max), (y_min, y_max)) in self.rooms.items():
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            canvas_x = (center_y + 1) * self.cell_size + self.cell_size // 2
            canvas_y = (center_x + 1) * self.cell_size + self.cell_size // 2

            self.canvas.create_text(canvas_x, canvas_y, text=room_name, fill="green")

    def is_room_full(self, room):
        x_range, y_range = self.rooms[room]

        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                pos = (x, y)
                if self.shelves.get(pos) and not any(p.position == pos for p in self.packages):
                    return False

        return True

    def assign_room_position(self, position, goods_type):
        goods_type = goods_type.lower()
        goods_type_encoded = self.goods_encoder.transform([goods_type])[0]
        x, y = position

        features = pd.DataFrame([[x, y, goods_type_encoded]], columns=['x', 'y', 'goods_type_encoded'])
        top_rooms = self.room_model.predict_proba(features)[0]
        top_rooms = list(np.argsort(top_rooms)[::-1])

        for i in top_rooms:
            assigned_room = self.room_encoder.inverse_transform([i])[0]

            if self.is_room_full(assigned_room):
                continue

            x_range, y_range = self.rooms[assigned_room]

            while True:
                assigned_x = random.randint(x_range[0], x_range[1])
                assigned_y = random.randint(y_range[0], y_range[1])

                if self.shelves.get((assigned_x, assigned_y)) and not any(
                        p.position == (assigned_x, assigned_y) for p in self.packages):
                    print(f"Paczka typu '{goods_type}' przydzielona do regału w pokoju '{assigned_room}'")
                    return (assigned_x, assigned_y), assigned_room

    def train_room_classifier(self):
        data = pd.read_csv("dataset_room.csv")

        self.goods_encoder = LabelEncoder()
        data['goods_type_encoded'] = self.goods_encoder.fit_transform(data['goods_type'])

        self.room_encoder = LabelEncoder()
        data['room_encoded'] = self.room_encoder.fit_transform(data['room'])

        X = data[['x', 'y', 'goods_type_encoded']]
        y = data['room_encoded']

        self.room_model = DecisionTreeClassifier()
        self.room_model.fit(X, y)

    def model_setup(self):
        data = pd.read_csv("dataset.csv")

        obstacle_encoder = LabelEncoder()
        data.obstacle_front = obstacle_encoder.fit_transform(data.obstacle_front)  # zamieniamy na liczby

        data["holding_package"] = data["holding_package"].astype(int)  # zamieniamy na liczby
        X = data[["x", "y", "direction", "battery", "obstacle_front", "is_wall_front", "dx_to_target", "dy_to_target",
                  "holding_package"]]

        y = data.action
        action_encoder = LabelEncoder()
        y = action_encoder.fit_transform(y)  # zamieniamy na liczby

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        # zapisujemy model i encodery do predykcji
        self.forklift.model = model
        self.forklift.obstacle_encoder = obstacle_encoder
        self.forklift.action_encoder = action_encoder

    def move_to_target(self, target):
        if target != self.charging_station:
            to_target = self.estimate_battery_cost(target)
            to_charge = self.estimate_battery_cost(
                self.charging_station,
                start_pos=target,
                start_dir=self.forklift.direction
            )
            total_needed = to_target + to_charge + 20  # dla bezpieczeństwa

            if total_needed > self.forklift.battery_level:
                print(
                    f"Potrzebujesz {total_needed}% baterii (cel+powrót), masz {self.forklift.battery_level}%. Najpierw tankuję.")
                self.move_to_target(self.charging_station)

        if self.forklift.position == target:
            return
        # zaznaczamy cel na czerwono
        if self.current_target:
            self.canvas.delete(self.current_target)

        r, c = target
        x1, y1 = (c + 1) * self.cell_size, (r + 1) * self.cell_size
        x2, y2 = x1 + self.cell_size, y1 + self.cell_size
        self.current_target = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3)

        path = self.forklift.astar(target)
        if not path:
            print("No path found")
            return

        for action in path:
            # possible actions: go forward, turn left, turn right
            if action == "Forward":
                self.forklift.move_forward()
            elif action == "Left":
                self.forklift.rotate_left()
            elif action == "Right":
                self.forklift.rotate_right()

            self.root.update()
            self.root.after(80)  # opóźnienie dla wizualizacji
        self.update_battery_label()

    def estimate_battery_cost(self, target, start_pos=None, start_dir=None):
        """
        Zwraca ile % baterii potrzeba, by z (start_pos, start_dir) dojechać do target.
        Jeśli nie podano start_pos/start_dir, używa aktualnego stanu wózka.
        """
        # 1) zapamiętaj oryginalny stan
        orig_pos = self.forklift.position
        orig_dir = self.forklift.direction

        # 2) ewentualnie ustaw stan startowy
        if start_pos is not None:
            self.forklift.position = start_pos
        if start_dir is not None:
            self.forklift.direction = start_dir

        # 3) wylicz ścieżkę i koszt (tak jak masz teraz)
        path = self.forklift.astar(target)
        cost = 0
        if path is None:
            needed = float('inf')
        else:
            pos = self.forklift.position
            dir = self.forklift.direction
            for action in path:
                if action == "Forward":
                    # oblicz nową pozycję + koszt kałuża/dziura/normal
                    x, y = pos
                    if dir == 0:
                        y += 1
                    elif dir == 90:
                        x -= 1
                    elif dir == 180:
                        y -= 1
                    elif dir == 270:
                        x += 1
                    pos = (x, y)
                    obst = self.obstacles.get(pos)
                    cost += HOLE_COST if obst == "hole" else PUDDLE_COST if obst == "puddle" else 1
                else:
                    cost += 1
                    if action == "Left":
                        dir = (dir + 90) % 360
                    else:
                        dir = (dir - 90) % 360
            needed = cost

        # 4) przywróć stan wózka
        self.forklift.position = orig_pos
        self.forklift.direction = orig_dir

        return needed

    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = (c + 1) * self.cell_size, (r + 1) * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")

    def draw_charging_station(self):
        r, c = self.charging_station
        x, y = (c + 1) * self.cell_size, (r + 1) * self.cell_size
        self.charging_image = Image.open("bateria.png")
        self.charging_image = self.charging_image.resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.charging_tk_image = ImageTk.PhotoImage(self.charging_image)
        self.canvas.create_image(x, y, anchor="nw", image=self.charging_tk_image)

    def create_package(self, position):
        files = [f for f in os.listdir("packages") if f.lower().endswith(".png")]

        filename = random.choice(files)
        filepath = os.path.join("packages", filename)

        package = Package(position, goods_type=None)
        package.image_path = filepath
        package.image = Image.open("package.png")
        image = package.image.resize((self.cell_size, self.cell_size), Image.LANCZOS)
        package.tk_image = ImageTk.PhotoImage(image)
        r, c = position
        x, y = (c + 1) * self.cell_size, (r + 1) * self.cell_size
        package.canvas_image_id = self.canvas.create_image(x, y, anchor="nw", image=package.tk_image)
        self.packages.append(package)

    def create_shelves(self):
        image = Image.open("regal.png").resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.shelf_tk_image = ImageTk.PhotoImage(image)
        if not hasattr(self.canvas, "image_refs"):
            self.canvas.image_refs = []
        self.canvas.image_refs.append(self.shelf_tk_image)
        for pos in self.shelf_positions:
            self.shelves[pos] = Shelf(pos)
            self.draw_shelf(pos)

        # za ładowaniem obrazka regału
        self.sciana_image = Image.open("sciana.jpg") \
            .resize((self.cell_size, self.cell_size), Image.LANCZOS)
        self.sciana_tk = ImageTk.PhotoImage(self.sciana_image)
        self.canvas.image_refs.append(self.sciana_tk)

        # rysujemy każdą ścianę
        for pos in self.sciany:
            r, c = pos
            x, y = (c + 1) * self.cell_size, (r + 1) * self.cell_size
            self.canvas.create_image(x, y, anchor="nw", image=self.sciana_tk)

    def draw_shelf(self, position):
        r, c = position
        x = (c + 1) * self.cell_size
        y = (r + 1) * self.cell_size
        self.canvas.create_image(x, y, anchor="nw", image=self.shelf_tk_image)

    def draw_obstacles(self):
        for (r, c), obstacle_type in self.obstacles.items():
            x = (c + 1) * self.cell_size
            y = (r + 1) * self.cell_size
            if obstacle_type == "puddle":
                self.canvas.create_image(x, y, anchor="nw", image=self.puddle_tk_image)
            elif obstacle_type == "hole":
                self.canvas.create_image(x, y, anchor="nw", image=self.hole_tk_image)

    def draw_forklift(self):
        self.forklift.tk_image = self.forklift.get_rotated_image()
        r, c = self.forklift.position
        x, y = (c + 1) * self.cell_size, (r + 1) * self.cell_size
        self.forklift.canvas_image = self.canvas.create_image(x, y, anchor="nw", image=self.forklift.tk_image)

    def move_forklift(self):
        r, c = self.forklift.position
        x1, y1 = (c + 1) * self.cell_size, (r + 1) * self.cell_size
        self.canvas.coords(self.forklift.canvas_image, x1, y1)
        if self.forklift.position == self.charging_station:
            self.forklift.battery_level = 200  # ładowanie
        self.update_battery_label()

    def update_battery_label(self):
        self.battery_label.config(text=f"Battery: {self.forklift.battery_level // 2}%")


class Forklift:
    def __init__(self, app):
        self.position = (5, 5)
        self.direction = 0  # 0: right, 90: up, 180: left, 270: down
        self.original_image = Image.open("wozek.png")
        self.tk_image = None
        self.battery_level = 200
        self.app = app
        self.holding_package = None
        self.holding_packages = []

        self.canvas_image = None
        self.model = None
        self.obstacle_encoder = None
        self.action_encoder = None

    def get_rotated_image(self):
        rotated = self.original_image.rotate(self.direction, expand=True)
        resized = rotated.resize((self.app.cell_size, self.app.cell_size), Image.LANCZOS)
        return ImageTk.PhotoImage(resized)

    def rotate_left(self):
        if self.battery_level <= 0:
            return
        self.direction = (self.direction + 90) % 360
        self.tk_image = self.get_rotated_image()
        self.app.canvas.itemconfig(self.canvas_image, image=self.tk_image)
        self.decrease_battery(self.position)

    def rotate_right(self):
        if self.battery_level <= 0:
            return
        self.direction = (self.direction - 90) % 360
        self.tk_image = self.get_rotated_image()
        self.app.canvas.itemconfig(self.canvas_image, image=self.tk_image)
        self.decrease_battery(self.position)

    def move_forward(self):
        if self.battery_level <= 0:
            return
        x, y = self.position
        dx, dy = 0, 0
        if self.direction == 0:
            dy = 1  # Right
        elif self.direction == 90:
            dx = -1  # Up
        elif self.direction == 180:
            dy = -1  # Left
        elif self.direction == 270:
            dx = 1  # Down
        new_x, new_y = x + dx, y + dy

        if (0 <= new_x < self.app.rows
                and 0 <= new_y < self.app.cols
                and (new_x, new_y) not in self.app.shelf_positions
                and (new_x, new_y) not in self.app.sciany):
            # dopiero gdy nie ma ani regału, ani ściany
            self.position = (new_x, new_y)
            self.decrease_battery(self.position)
            self.app.move_forklift()
            self.app.update_battery_label()

    def decrease_battery(self, position):
        obstacle_type = self.app.obstacles.get(position)
        if obstacle_type == "hole":
            self.battery_level = max(0, self.battery_level - HOLE_COST)
        elif obstacle_type == "puddle":
            self.battery_level = max(0, self.battery_level - PUDDLE_COST)
        else:
            self.battery_level = max(0, self.battery_level - 1)
        self.app.update_battery_label()

    def get_front_position(self):
        x, y = self.position
        if self.direction == 0:
            return (x, y + 1)
        elif self.direction == 90:
            return (x - 1, y)
        elif self.direction == 180:
            return (x, y - 1)
        elif self.direction == 270:
            return (x + 1, y)

    def user_move(self, event):
        if event.keysym == 'Up':
            self.move_forward()
        elif event.keysym == 'Left':
            self.rotate_left()
        elif event.keysym == 'Right':
            self.rotate_right()
        elif event.char.lower() == 'p':
            self.pickup()
        elif event.char.lower() == 'd':
            self.drop()

    def pickup(self):
        front_pos = self.get_front_position()
        for package in self.app.packages:
            if package.position == front_pos:
                # classify on pickup
                if package.goods_type is None:
                    digit = self.app.classifier.predict_digit(package.image_path)
                    digit_to_goods_type = {
                        0: "goods_type_a",
                        1: "goods_type_b",
                        2: "goods_type_c",
                        3: "goods_type_d",
                        4: "goods_type_e",
                        5: "goods_type_f",
                        6: "goods_type_g",
                        7: "goods_type_h"
                    }
                    package.goods_type = digit_to_goods_type.get(digit, f"goods_type_{digit}")
                    print(f"[PACZKA] Rozpoznano cyfrę: {digit} → przypisano goods_type='{package.goods_type}'")

                self.holding_packages.append(package)
                self.app.canvas.delete(package.canvas_image_id)
                self.app.packages.remove(package)
                print(f"Picked up package: {package.goods_type}")
                return

    def drop(self, package):
        front_pos = self.get_front_position()
        if front_pos in self.app.shelf_positions:
            shelf = self.app.shelves.get(front_pos)
            if shelf:
                shelf.add_package(package)
                if package in self.holding_packages:
                    self.holding_packages.remove(package)
                self.app.create_package(front_pos)

    def heuristic(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def astar(self, goal):
        start_state = (self.position, self.direction)
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0, counter, Node(None, start_state, None, 0)))
        counter += 1
        visited = {}

        while frontier:
            _, _, current_node = heapq.heappop(frontier)
            if self.test_celu(current_node, goal):
                path = []
                while current_node.parent:
                    path.append(current_node.action)
                    current_node = current_node.parent
                path.reverse()
                return path

            state = current_node.state
            if state in visited and visited[state] <= current_node.cost:
                continue
            visited[state] = current_node.cost

            for next_state, action in self.funkcja_nastepnika(state, goal):
                position = next_state[0]
                obstacle_type = self.app.obstacles.get(position)
                move_cost = HOLE_COST if obstacle_type == "hole" else PUDDLE_COST if obstacle_type == "puddle" else 1
                new_cost = current_node.cost + move_cost
                h = self.heuristic(position, goal)
                f = new_cost + h
                child_node = Node(current_node, next_state, action, new_cost)
                heapq.heappush(frontier, (f, counter, child_node))
                counter += 1
        return None

    def funkcja_nastepnika(self, state, goal):
        (x, y), dir = state
        results = []

        # Try to move forward
        dx, dy = 0, 0
        if dir == 0:
            dy = 1
        elif dir == 90:
            dx = -1
        elif dir == 180:
            dy = -1
        elif dir == 270:
            dx = 1
        new_x, new_y = x + dx, y + dy

        front_pos = (new_x, new_y)
        if 0 <= new_x < self.app.rows and 0 <= new_y < self.app.cols:
            if front_pos not in self.app.shelf_positions and front_pos not in self.app.sciany:
                results.append((((new_x, new_y), dir), "Forward"))
            else:
                # ZMIANA: użycie modelu ML gdy przed nami przeszkoda
                model_action = self.predict_action((x, y), dir, front_pos, goal)
                results.append((((x, y), self.rotate(dir, model_action)), model_action))
        else:
            model_action = self.predict_action((x, y), dir, front_pos, goal)
            results.append((((x, y), self.rotate(dir, model_action)), model_action))

        # Default rotations
        results.append((((x, y), (dir + 90) % 360), "Left"))
        results.append((((x, y), (dir - 90) % 360), "Right"))
        return results

    def rotate(self, direction, action):
        if action == "Left":
            return (direction + 90) % 360
        elif action == "Right":
            return (direction - 90) % 360
        return direction

    def predict_action(self, pos, direction, front_pos, goal):
        if not self.model:
            return "Left"  # fallback
        x, y = pos
        dx = goal[0] - x
        dy = goal[1] - y
        obstacle = self.app.obstacles.get(front_pos, "none")
        obstacle_encoded = self.obstacle_encoder.transform([obstacle])[0] if obstacle != "none" else 0
        is_wall_front = int(front_pos in self.app.shelf_positions)
        data = pd.DataFrame([{
            "x": x,
            "y": y,
            "direction": direction,
            "battery": self.battery_level,
            "obstacle_front": obstacle_encoded,
            "is_wall_front": is_wall_front,
            "dx_to_target": dx,
            "dy_to_target": dy,
            "holding_package": int(self.holding_package is not None)
        }])
        action_encoded = self.model.predict(data)[0]
        action = self.action_encoder.inverse_transform([action_encoded])[0]
        return action

    def test_celu(self, node, goal):
        return node.state[0] == goal


class Node:
    def __init__(self, parent, state, action, cost=0):
        self.parent = parent
        self.state = state
        self.action = action
        self.cost = cost  # g(n)


class Package:
    def __init__(self, position, goods_type):
        self.position = position
        self.goods_type = goods_type
        self.image = None
        self.tk_image = None
        self.canvas_image_id = None  # ID obrazka na canvasie


class Shelf:
    def __init__(self, position):
        self.position = position
        self.packages = []

    def add_package(self, package):
        self.packages.append(package)
        return True

    def remove_package(self):
        if self.packages:
            return self.packages.pop()
        return None


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Symulacja wózka widłowego")
    app = GridApp(root)
    root.mainloop()
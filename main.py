"""
AI GAME GENERATOR with Groq API
Стабильная версия - исправлены все импорты
"""

import json
import random
import math
import time
import os
import re

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition, SlideTransition
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Ellipse, Line, Triangle
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.properties import NumericProperty

# Безопасное определение платформы
try:
    from kivy.utils import platform
except:
    platform = 'unknown'

# Безопасный импорт requests
HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except:
    pass


# ==================== КОНФИГУРАЦИЯ ====================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "recommended": True, "speed": "Fast", "quality": "Best"},
    {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B", "recommended": True, "speed": "Fast", "quality": "Excellent"},
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "recommended": True, "speed": "Ultra Fast", "quality": "Good"},
    {"id": "llama3-70b-8192", "name": "Llama 3 70B", "recommended": False, "speed": "Fast", "quality": "Great"},
    {"id": "llama3-8b-8192", "name": "Llama 3 8B", "recommended": False, "speed": "Very Fast", "quality": "Good"},
    {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "recommended": False, "speed": "Fast", "quality": "Great"},
]

GAME_PROMPT = """Create a mini-game config. Return ONLY JSON:
{{"name": "Game Name", "theme": "space", "game_type": "shooter", "description": "Desc",
"player": {{"shape": "triangle", "color": [0.3, 0.7, 1.0], "speed": 5, "health": 3}},
"enemies": [{{"name": "Enemy", "shape": "circle", "color": [1, 0.3, 0.3], "speed": 1.5, "health": 1, "points": 10, "behavior": "chase"}}],
"items": [{{"name": "Health", "color": [0.3, 1, 0.3], "effect": "heal", "value": 1}}],
"background_colors": [[0.1, 0.1, 0.2]]}}

User idea: {prompt}"""

LOCAL_THEMES = {
    'space': {'bg': [[0.05, 0.05, 0.15]], 'player': [0.3, 0.8, 1], 'enemies': [[1, 0.3, 0.3], [1, 0.5, 0.2]], 'names': ['Cosmic', 'Stellar', 'Galaxy']},
    'fantasy': {'bg': [[0.1, 0.15, 0.1]], 'player': [0.9, 0.7, 0.3], 'enemies': [[0.5, 0.2, 0.5], [0.3, 0.5, 0.2]], 'names': ['Dragon', 'Magic', 'Quest']},
    'cyber': {'bg': [[0.02, 0.02, 0.08]], 'player': [0, 1, 1], 'enemies': [[1, 0, 0.5], [0.5, 0, 1]], 'names': ['Neon', 'Cyber', 'Matrix']},
    'ocean': {'bg': [[0.02, 0.1, 0.2]], 'player': [0.2, 0.8, 0.7], 'enemies': [[0.5, 0.3, 0.5], [0.6, 0.2, 0.3]], 'names': ['Ocean', 'Aqua', 'Deep']},
}

FUN_FACTS = ["Connecting to AI...", "Creating game...", "Designing enemies...", "Adding power-ups...", "Almost ready..."]


# ==================== УТИЛИТЫ ====================

def get_data_path():
    """Безопасное получение пути для данных"""
    try:
        if platform == 'android':
            # Пробуем разные варианты
            try:
                from android.storage import app_storage_path
                return app_storage_path()
            except:
                pass
            # Fallback пути
            for path in ['/data/data/org.groq.aigamegen/files', '/sdcard/AIGameGen']:
                try:
                    os.makedirs(path, exist_ok=True)
                    return path
                except:
                    continue
    except:
        pass
    return os.path.expanduser('~')


def check_internet():
    """Простая проверка интернета"""
    if not HAS_REQUESTS:
        return False
    try:
        requests.get("https://api.groq.com", timeout=5)
        return True
    except:
        return False


# ==================== GROQ API ====================

class GroqClient:
    def __init__(self):
        self.api_key = ""
        self.model = "llama-3.1-8b-instant"
    
    def set_api_key(self, key):
        self.api_key = key.strip() if key else ""
    
    def set_model(self, model_id):
        self.model = model_id
    
    def generate_game(self, prompt):
        if not HAS_REQUESTS:
            return False, "No internet library"
        if not self.api_key:
            return False, "No API key"
        
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": GAME_PROMPT.format(prompt=prompt)}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 401:
                return False, "Invalid API Key"
            if response.status_code != 200:
                return False, f"Error: {response.status_code}"
            
            content = response.json()['choices'][0]['message']['content']
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return True, json.loads(match.group())
            return False, "Invalid response"
            
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, "No connection"
        except Exception as e:
            return False, str(e)[:30]
    
    def test_connection(self):
        if not HAS_REQUESTS:
            return False, "No internet library"
        if not self.api_key:
            return False, "Enter API key"
        
        try:
            response = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5},
                timeout=10
            )
            if response.status_code == 401:
                return False, "Invalid API Key"
            if response.status_code == 200:
                return True, "Connected!"
            return False, f"Error: {response.status_code}"
        except:
            return False, "Connection failed"


def generate_local_game(prompt):
    """Локальная генерация"""
    prompt_lower = prompt.lower()
    
    theme = 'space'
    for t in LOCAL_THEMES:
        if t in prompt_lower:
            theme = t
            break
    
    data = LOCAL_THEMES[theme]
    name = f"{random.choice(['Super', 'Ultra', 'Mega'])} {random.choice(data['names'])} {random.choice(['Quest', 'Rush', 'Blitz'])}"
    
    enemies = []
    for i, color in enumerate(data['enemies']):
        enemies.append({
            'name': f"Enemy {i+1}", 'shape': 'circle', 'color': color,
            'speed': 1 + i * 0.5, 'health': 1 + i, 'points': 10 + i * 10,
            'behavior': ['chase', 'patrol', 'zigzag'][i % 3]
        })
    
    return {
        'name': name, 'theme': theme, 'game_type': 'shooter',
        'description': f"Local {theme} game",
        'player': {'shape': 'triangle', 'color': data['player'], 'speed': 5, 'health': 3},
        'enemies': enemies,
        'items': [
            {'name': 'Health', 'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1},
            {'name': 'Shield', 'color': [0.3, 0.7, 1], 'effect': 'shield', 'value': 3},
        ],
        'background_colors': data['bg'],
        'created_at': time.time(),
        'ai_generated': False
    }


# ==================== ИГРОВЫЕ ОБЪЕКТЫ ====================

class Player:
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.size = dp(30)
        self.color = tuple(config.get('color', [0.3, 0.7, 1]))
        self.shape = config.get('shape', 'triangle')
        self.max_health = config.get('health', 3)
        self.health = self.max_health
        self.speed = config.get('speed', 5)
        self.shield = 0
        self.invincible = 0
        self.trail = []
    
    def update(self, dt):
        self.shield = max(0, self.shield - dt)
        self.invincible = max(0, self.invincible - dt)
        self.trail.append((self.x, self.y, time.time()))
        self.trail = [(x, y, t) for x, y, t in self.trail if time.time() - t < 0.3]
    
    def take_damage(self):
        if self.invincible > 0 or self.shield > 0:
            self.shield = max(0, self.shield - 1)
            return False
        self.health -= 1
        self.invincible = 1.5
        return self.health <= 0


class Enemy:
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.size = dp(25)
        self.color = tuple(config.get('color', [1, 0.3, 0.3]))
        self.shape = config.get('shape', 'circle')
        self.speed = config.get('speed', 1)
        self.health = config.get('health', 1)
        self.max_health = self.health
        self.points = config.get('points', 10)
        self.behavior = config.get('behavior', 'chase')
        self.timer = 0
        self.vx = 0
        self.vy = 0
    
    def update(self, dt, tx=None, ty=None):
        self.timer += dt
        if self.behavior == 'chase' and tx is not None:
            dx, dy = tx - self.x, ty - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                self.vx = (dx / dist) * self.speed * dp(60)
                self.vy = (dy / dist) * self.speed * dp(60)
        elif self.behavior == 'patrol':
            self.vx = math.sin(self.timer * 2) * self.speed * dp(60)
            self.vy = -self.speed * dp(30)
        elif self.behavior == 'zigzag':
            self.vx = math.sin(self.timer * 5) * self.speed * dp(80)
            self.vy = -self.speed * dp(40)
        else:
            self.vy = -self.speed * dp(40)
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def collides_with(self, other):
        dist = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        return dist < (self.size + other.size) / 2


class Item:
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.size = dp(20)
        self.color = tuple(config.get('color', [1, 1, 0.3]))
        self.effect = config.get('effect', 'heal')
        self.value = config.get('value', 1)
        self.pulse = 0
        self.vy = -dp(30)
    
    def update(self, dt):
        self.pulse += dt * 5
        self.y += self.vy * dt
    
    def collides_with(self, other):
        dist = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        return dist < (self.size + other.size) / 2


class Bullet:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.size = dp(8)
        self.color = color
        self.vy = dp(400)
    
    def update(self, dt):
        self.y += self.vy * dt
    
    def collides_with(self, other):
        dist = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        return dist < (self.size + other.size) / 2


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-100, 100)
        self.vy = random.uniform(-100, 100)
        self.life = 1.0
        self.size = random.uniform(3, 8)
    
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt * 2
        self.size *= 0.95
        return self.life > 0


# ==================== ИГРОВОЙ ДВИЖОК ====================

class GameEngine(Widget):
    score = NumericProperty(0)
    level = NumericProperty(1)
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.player = None
        self.enemies = []
        self.items = []
        self.bullets = []
        self.particles = []
        self.game_time = 0
        self.spawn_timer = 0
        self.enemies_killed = 0
        self.items_collected = 0
        self.combo = 0
        self.combo_timer = 0
        self.max_combo = 0
        self.paused = False
        self.game_over = False
        self.touching = False
        self.touch_pos = (0, 0)
        self.stars = []
        self.bind(size=self.setup_stars)
    
    def setup_stars(self, *args):
        w = max(100, self.width)
        h = max(100, self.height)
        self.stars = [{'x': random.uniform(0, w), 'y': random.uniform(0, h),
                       'size': random.uniform(1, 3), 'speed': random.uniform(20, 80),
                       'b': random.uniform(0.3, 1.0)} for _ in range(40)]
    
    def start_game(self):
        self.setup_stars()
        cfg = self.config.get('player', {})
        self.player = Player(self.width / 2, self.height * 0.2, cfg)
        self.enemies = []
        self.items = []
        self.bullets = []
        self.particles = []
        self.score = 0
        self.level = 1
        self.game_time = 0
        self.spawn_timer = 0
        self.enemies_killed = 0
        self.items_collected = 0
        self.combo = 0
        self.max_combo = 0
        self.paused = False
        self.game_over = False
    
    def update(self, dt):
        if self.paused or self.game_over or not self.player:
            self.draw()
            return
        
        self.game_time += dt
        self.spawn_timer += dt
        self.combo_timer -= dt
        if self.combo_timer <= 0:
            self.combo = 0
        
        # Level up
        if self.enemies_killed >= 5 + self.level * 3 and not self.enemies:
            self.level += 1
            self.enemies_killed = 0
            self.score += self.level * 100
        
        # Controls
        if self.touching:
            tx, ty = self.touch_pos
            dx, dy = tx - self.player.x, ty - self.player.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 5:
                speed = self.player.speed * dp(60) * dt
                self.player.x += (dx / dist) * min(speed, dist)
                self.player.y += (dy / dist) * min(speed, dist)
        
        # Boundaries
        self.player.x = max(self.player.size/2, min(self.width - self.player.size/2, self.player.x))
        self.player.y = max(self.player.size/2, min(self.height - self.player.size/2, self.player.y))
        self.player.update(dt)
        
        # Auto-shoot
        if self.config.get('game_type') == 'shooter':
            if int(self.game_time * 5) % 2 == 0 and len(self.bullets) < 10:
                self.bullets.append(Bullet(self.player.x, self.player.y + self.player.size/2, self.player.color))
        
        # Spawn
        spawn_rate = 2.0 / (1 + self.level * 0.1)
        if self.spawn_timer > spawn_rate and len(self.enemies) < 10:
            self.spawn_enemy()
            self.spawn_timer = 0
        if random.random() < 0.005:
            self.spawn_item()
        
        # Update enemies
        for e in self.enemies[:]:
            e.update(dt, self.player.x, self.player.y)
            if e.y < -e.size or e.y > self.height + e.size:
                self.enemies.remove(e)
            elif e.collides_with(self.player):
                if self.player.take_damage():
                    self.game_over = True
                else:
                    self.spawn_particles(e.x, e.y, e.color)
                    self.enemies.remove(e)
        
        # Update items
        for i in self.items[:]:
            i.update(dt)
            if i.y < -i.size:
                self.items.remove(i)
            elif i.collides_with(self.player):
                self.collect_item(i)
                self.items.remove(i)
        
        # Update bullets
        for b in self.bullets[:]:
            b.update(dt)
            if b.y > self.height + b.size:
                self.bullets.remove(b)
            else:
                for e in self.enemies[:]:
                    if b.collides_with(e):
                        e.health -= 1
                        if e.health <= 0:
                            self.kill_enemy(e)
                        if b in self.bullets:
                            self.bullets.remove(b)
                        self.spawn_particles(b.x, b.y, b.color)
                        break
        
        # Update particles
        self.particles = [p for p in self.particles if p.update(dt)]
        
        # Update stars
        for s in self.stars:
            s['y'] -= s['speed'] * dt
            if s['y'] < 0:
                s['y'] = self.height
                s['x'] = random.uniform(0, self.width)
        
        self.draw()
    
    def spawn_enemy(self):
        configs = self.config.get('enemies', [{'color': [1, 0.3, 0.3], 'speed': 1, 'health': 1, 'points': 10}])
        cfg = random.choice(configs).copy()
        cfg['speed'] = cfg.get('speed', 1) * (1 + self.level * 0.1)
        x = random.uniform(dp(30), self.width - dp(30))
        self.enemies.append(Enemy(x, self.height + dp(30), cfg))
    
    def spawn_item(self):
        configs = self.config.get('items', [{'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1}])
        cfg = random.choice(configs)
        x = random.uniform(dp(30), self.width - dp(30))
        self.items.append(Item(x, self.height + dp(20), cfg))
    
    def kill_enemy(self, e):
        pts = e.points
        self.combo += 1
        self.combo_timer = 2.0
        self.max_combo = max(self.max_combo, self.combo)
        self.score += int(pts * (1 + self.combo * 0.1))
        self.enemies_killed += 1
        self.spawn_particles(e.x, e.y, e.color)
        if e in self.enemies:
            self.enemies.remove(e)
    
    def collect_item(self, i):
        self.items_collected += 1
        self.score += 25
        if i.effect == 'heal':
            self.player.health = min(self.player.max_health, self.player.health + i.value)
        elif i.effect == 'shield':
            self.player.shield += i.value
        self.spawn_particles(i.x, i.y, i.color)
    
    def spawn_particles(self, x, y, color):
        for _ in range(8):
            self.particles.append(Particle(x, y, color))
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.touching = True
            self.touch_pos = touch.pos
            return True
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.touching:
            self.touch_pos = touch.pos
            return True
        return super().on_touch_move(touch)
    
    def on_touch_up(self, touch):
        self.touching = False
        return super().on_touch_up(touch)
    
    def draw(self):
        self.canvas.clear()
        bg = self.config.get('background_colors', [[0.1, 0.1, 0.2]])[0]
        
        with self.canvas:
            Color(*bg, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            for s in self.stars:
                Color(s['b'], s['b'], s['b'] * 1.1, s['b'])
                Ellipse(pos=(s['x'], s['y']), size=(s['size'], s['size']))
            
            for p in self.particles:
                Color(*p.color, p.life)
                Ellipse(pos=(p.x - p.size/2, p.y - p.size/2), size=(p.size, p.size))
            
            for i in self.items:
                pulse = 1 + 0.2 * math.sin(i.pulse)
                sz = i.size * pulse
                Color(*i.color, 0.3)
                Ellipse(pos=(i.x - sz, i.y - sz), size=(sz * 2, sz * 2))
                Color(*i.color, 1)
                Ellipse(pos=(i.x - i.size/2, i.y - i.size/2), size=(i.size, i.size))
            
            for b in self.bullets:
                Color(*b.color, 1)
                Ellipse(pos=(b.x - b.size/2, b.y - b.size/2), size=(b.size, b.size))
            
            for e in self.enemies:
                Color(0, 0, 0, 0.3)
                Ellipse(pos=(e.x - e.size/2 + 3, e.y - e.size/2 - 3), size=(e.size, e.size))
                Color(*e.color, 1)
                Ellipse(pos=(e.x - e.size/2, e.y - e.size/2), size=(e.size, e.size))
            
            if self.player:
                for idx, (tx, ty, _) in enumerate(self.player.trail):
                    alpha = idx / max(1, len(self.player.trail)) * 0.3
                    Color(*self.player.color, alpha)
                    sz = self.player.size * 0.5
                    Ellipse(pos=(tx - sz/2, ty - sz/2), size=(sz, sz))
                
                Color(*self.player.color, 0.2)
                glow = self.player.size * 1.5
                Ellipse(pos=(self.player.x - glow/2, self.player.y - glow/2), size=(glow, glow))
                
                if self.player.invincible > 0 and int(self.player.invincible * 10) % 2:
                    Color(*self.player.color, 0.5)
                else:
                    Color(*self.player.color, 1)
                
                h = self.player.size / 2
                Triangle(points=[self.player.x, self.player.y + h, 
                                self.player.x - h, self.player.y - h,
                                self.player.x + h, self.player.y - h])
                
                if self.player.shield > 0:
                    Color(0.3, 0.7, 1, 0.4)
                    Line(circle=(self.player.x, self.player.y, self.player.size * 0.8), width=2)


# ==================== ЭКРАНЫ ====================

class BaseScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.05, 0.05, 0.1, 1)
            self.bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_bg, size=self.update_bg)
    
    def update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size


class SettingsScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.selected_model = GROQ_MODELS[0]['id']
        
        layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(6))
        
        # Header
        header = BoxLayout(size_hint=(1, 0.07))
        back = Button(text='<', size_hint=(0.15, 1), font_size=sp(20),
                     background_color=(0.3, 0.3, 0.4, 1), background_normal='')
        back.bind(on_release=lambda x: self.go_back())
        header.add_widget(back)
        header.add_widget(Label(text='Settings', font_size=sp(18), color=(0.8, 0.8, 1, 1)))
        layout.add_widget(header)
        
        # Status
        self.status_lbl = Label(text='', font_size=sp(11), size_hint=(1, 0.04), color=(0.5, 0.5, 0.5, 1))
        layout.add_widget(self.status_lbl)
        
        # API Key
        layout.add_widget(Label(text='Groq API Key:', font_size=sp(13), size_hint=(1, 0.04), color=(0.5, 0.8, 1, 1)))
        
        self.api_input = TextInput(hint_text='gsk_xxxxx...', multiline=False, password=True,
                                   font_size=sp(13), size_hint=(1, 0.08),
                                   background_color=(0.12, 0.12, 0.18, 1), foreground_color=(1, 1, 1, 1))
        key = self.app.load_api_key()
        if key:
            self.api_input.text = key
        layout.add_widget(self.api_input)
        
        # Buttons
        btns = BoxLayout(size_hint=(1, 0.07), spacing=dp(8))
        
        show = Button(text='Show', font_size=sp(11), background_color=(0.3, 0.3, 0.4, 1), background_normal='')
        show.bind(on_release=lambda x: self.toggle_pass())
        btns.add_widget(show)
        
        test = Button(text='Test', font_size=sp(11), background_color=(0.3, 0.5, 0.6, 1), background_normal='')
        test.bind(on_release=lambda x: self.test_api())
        btns.add_widget(test)
        
        save = Button(text='Save', font_size=sp(11), background_color=(0.3, 0.5, 0.3, 1), background_normal='')
        save.bind(on_release=lambda x: self.save_key())
        btns.add_widget(save)
        
        layout.add_widget(btns)
        
        layout.add_widget(Label(text='Get key: console.groq.com', font_size=sp(10),
                                size_hint=(1, 0.03), color=(0.4, 0.6, 0.8, 1)))
        
        # Models
        layout.add_widget(Label(text='Select Model:', font_size=sp(13), size_hint=(1, 0.04), color=(0.5, 0.8, 1, 1)))
        
        scroll = ScrollView(size_hint=(1, 0.45))
        self.grid = GridLayout(cols=1, spacing=dp(4), size_hint_y=None, padding=dp(3))
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.refresh_models()
        scroll.add_widget(self.grid)
        layout.add_widget(scroll)
        
        self.sel_lbl = Label(text=f'Selected: {GROQ_MODELS[0]["name"]}', font_size=sp(11),
                             size_hint=(1, 0.04), color=(0.5, 1, 0.5, 1))
        layout.add_widget(self.sel_lbl)
        
        self.add_widget(layout)
    
    def refresh_models(self):
        self.grid.clear_widgets()
        for m in GROQ_MODELS:
            sel = m['id'] == self.selected_model
            bg = (0.25, 0.35, 0.25, 1) if sel else ((0.2, 0.28, 0.2, 1) if m['recommended'] else (0.15, 0.15, 0.2, 1))
            prefix = "* " if m['recommended'] else "  "
            btn = Button(text=f"{prefix}{m['name']}\n   {m['speed']}",
                        font_size=sp(10), size_hint_y=None, height=dp(45),
                        background_color=bg, background_normal='')
            btn.bind(on_release=lambda x, mm=m: self.select_model(mm))
            self.grid.add_widget(btn)
    
    def select_model(self, m):
        self.selected_model = m['id']
        self.sel_lbl.text = f"Selected: {m['name']}"
        self.app.groq_client.set_model(m['id'])
        self.app.save_model(m['id'])
        self.refresh_models()
    
    def toggle_pass(self):
        self.api_input.password = not self.api_input.password
    
    def save_key(self):
        key = self.api_input.text.strip()
        self.app.save_api_key(key)
        self.app.groq_client.set_api_key(key)
        self.status_lbl.text = 'Saved!'
        self.status_lbl.color = (0.5, 1, 0.5, 1)
    
    def test_api(self):
        key = self.api_input.text.strip()
        if not key:
            self.status_lbl.text = 'Enter key first'
            self.status_lbl.color = (1, 0.5, 0.5, 1)
            return
        
        self.app.groq_client.set_api_key(key)
        self.status_lbl.text = 'Testing...'
        self.status_lbl.color = (1, 1, 0.5, 1)
        Clock.schedule_once(lambda dt: self._do_test(), 0.1)
    
    def _do_test(self):
        ok, msg = self.app.groq_client.test_connection()
        self.status_lbl.text = msg
        self.status_lbl.color = (0.5, 1, 0.5, 1) if ok else (1, 0.5, 0.5, 1)
    
    def go_back(self):
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'home'


class HomeScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        
        layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(6))
        
        layout.add_widget(Label(text='AI GAME', font_size=sp(34), size_hint=(1, 0.09), bold=True, color=(0.5, 0.8, 1, 1)))
        layout.add_widget(Label(text='GENERATOR', font_size=sp(26), size_hint=(1, 0.07), bold=True, color=(1, 0.5, 0.8, 1)))
        
        self.status_lbl = Label(text='', font_size=sp(10), size_hint=(1, 0.04), color=(0.5, 0.5, 0.5, 1))
        layout.add_widget(self.status_lbl)
        
        layout.add_widget(Label(text='Describe your game:', font_size=sp(12), size_hint=(1, 0.04), color=(0.7, 0.7, 0.8, 1)))
        
        self.prompt = TextInput(hint_text='Space shooter with aliens...', multiline=True,
                                font_size=sp(12), size_hint=(1, 0.15),
                                background_color=(0.1, 0.1, 0.15, 1), foreground_color=(1, 1, 1, 1))
        layout.add_widget(self.prompt)
        
        # Examples
        examples = GridLayout(cols=2, size_hint=(1, 0.09), spacing=dp(4))
        for name, prompt in [('Space', 'space shooter'), ('Fantasy', 'magic dragon'),
                             ('Ocean', 'underwater fish'), ('Cyber', 'neon robot')]:
            btn = Button(text=name, font_size=sp(10), background_color=(0.2, 0.2, 0.3, 1), background_normal='')
            btn.bind(on_release=lambda x, p=prompt: setattr(self.prompt, 'text', p))
            examples.add_widget(btn)
        layout.add_widget(examples)
        
        # Generate
        self.gen_btn = Button(text='GENERATE WITH AI', font_size=sp(16), size_hint=(1, 0.1),
                              background_color=(0.3, 0.6, 0.3, 1), background_normal='', bold=True)
        self.gen_btn.bind(on_release=lambda x: self.generate())
        layout.add_widget(self.gen_btn)
        
        # Offline
        self.local_btn = Button(text='Quick Generate (Offline)', font_size=sp(12), size_hint=(1, 0.07),
                                background_color=(0.4, 0.3, 0.5, 1), background_normal='')
        self.local_btn.bind(on_release=lambda x: self.gen_local())
        layout.add_widget(self.local_btn)
        
        # Bottom
        bottom = BoxLayout(size_hint=(1, 0.07), spacing=dp(8))
        settings = Button(text='Settings', font_size=sp(12), background_color=(0.35, 0.3, 0.45, 1), background_normal='')
        settings.bind(on_release=lambda x: self.go_settings())
        bottom.add_widget(settings)
        history = Button(text='History', font_size=sp(12), background_color=(0.25, 0.25, 0.35, 1), background_normal='')
        history.bind(on_release=lambda x: self.go_history())
        bottom.add_widget(history)
        layout.add_widget(bottom)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.update_status()
    
    def update_status(self):
        key = self.app.load_api_key()
        if key and HAS_REQUESTS:
            self.status_lbl.text = f'API: Ready | {self.app.groq_client.model}'
            self.status_lbl.color = (0.5, 1, 0.5, 1)
            self.gen_btn.background_color = (0.3, 0.6, 0.3, 1)
        elif not HAS_REQUESTS:
            self.status_lbl.text = 'Offline mode only'
            self.status_lbl.color = (1, 0.7, 0.3, 1)
            self.gen_btn.background_color = (0.4, 0.4, 0.4, 1)
        else:
            self.status_lbl.text = 'Set API key in Settings'
            self.status_lbl.color = (1, 0.5, 0.5, 1)
            self.gen_btn.background_color = (0.4, 0.4, 0.4, 1)
    
    def generate(self):
        key = self.app.load_api_key()
        if not key or not HAS_REQUESTS:
            self.status_lbl.text = 'Using offline mode...'
            Clock.schedule_once(lambda dt: self.gen_local(), 0.3)
            return
        
        self.app.current_prompt = self.prompt.text.strip() or "space shooter"
        self.manager.current = 'loading'
    
    def gen_local(self):
        prompt = self.prompt.text.strip() or "space shooter"
        self.app.current_game = generate_local_game(prompt)
        self.app.save_history(self.app.current_game)
        self.manager.current = 'game'
    
    def go_settings(self):
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'settings'
    
    def go_history(self):
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'history'


class LoadingScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.progress = 0
        self.api_done = False
        self.api_result = None
        self.api_error = None
        self.dots = 0
        
        layout = FloatLayout()
        
        self.title = Label(text='AI', font_size=sp(48), bold=True,
                          pos_hint={'center_x': 0.5, 'center_y': 0.7}, color=(0.5, 0.8, 1, 1))
        layout.add_widget(self.title)
        
        self.pbar = ProgressBar(max=100, value=0, size_hint=(0.8, 0.03),
                                pos_hint={'center_x': 0.5, 'center_y': 0.5})
        layout.add_widget(self.pbar)
        
        self.pct = Label(text='0%', font_size=sp(16), pos_hint={'center_x': 0.5, 'center_y': 0.45}, color=(0.5, 0.8, 1, 1))
        layout.add_widget(self.pct)
        
        self.status = Label(text='', font_size=sp(12), pos_hint={'center_x': 0.5, 'center_y': 0.38}, color=(0.6, 0.6, 0.7, 1))
        layout.add_widget(self.status)
        
        self.error = Label(text='', font_size=sp(10), pos_hint={'center_x': 0.5, 'center_y': 0.2}, color=(1, 0.5, 0.5, 1))
        layout.add_widget(self.error)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.progress = 0
        self.api_done = False
        self.api_result = None
        self.api_error = None
        self.dots = 0
        self.error.text = ''
        Clock.schedule_interval(self.update_ui, 0.05)
        Clock.schedule_once(self._call_api, 0.2)
    
    def _call_api(self, dt):
        ok, result = self.app.groq_client.generate_game(self.app.current_prompt)
        if ok:
            self.api_result = result
        else:
            self.api_error = result
        self.api_done = True
    
    def update_ui(self, dt):
        self.dots += dt
        
        if self.api_error:
            Clock.unschedule(self.update_ui)
            self.error.text = f"{self.api_error}\nUsing offline..."
            Clock.schedule_once(self._use_local, 1.5)
            return False
        
        if self.api_done and self.api_result:
            self.pbar.value = 100
            self.pct.text = "100%"
            self.status.text = "Ready!"
            Clock.unschedule(self.update_ui)
            Clock.schedule_once(self._finish, 0.5)
            return False
        
        if self.progress < 90:
            self.progress += random.uniform(0.5, 1.5)
        self.pbar.value = self.progress
        self.pct.text = f"{int(self.progress)}%"
        
        self.status.text = random.choice(FUN_FACTS) if int(self.dots) % 2 == 0 else self.status.text
        
        pulse = 1 + 0.1 * math.sin(self.dots * 5)
        self.title.font_size = sp(48 * pulse)
        
        return True
    
    def _use_local(self, dt):
        self.app.current_game = generate_local_game(self.app.current_prompt)
        self.app.save_history(self.app.current_game)
        self.manager.current = 'game'
    
    def _finish(self, dt):
        self.app.current_game = self.app.build_config(self.api_result)
        self.app.save_history(self.app.current_game)
        self.manager.current = 'game'
    
    def on_leave(self):
        Clock.unschedule(self.update_ui)


class GameScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.engine = None
        self.update_event = None
        self.layout = FloatLayout()
        self.add_widget(self.layout)
    
    def on_pre_enter(self):
        self.setup()
    
    def setup(self):
        self.layout.clear_widgets()
        if not self.app.current_game:
            return
        
        self.engine = GameEngine(self.app.current_game, size_hint=(1, 0.85))
        self.layout.add_widget(self.engine)
        
        # Top
        top = BoxLayout(size_hint=(1, 0.08), pos_hint={'top': 1}, padding=dp(5))
        with top.canvas.before:
            Color(0, 0, 0, 0.5)
            self.top_bg = Rectangle(pos=top.pos, size=top.size)
        top.bind(pos=lambda *x: setattr(self.top_bg, 'pos', top.pos), 
                size=lambda *x: setattr(self.top_bg, 'size', top.size))
        
        self.name_lbl = Label(text=self.app.current_game.get('name', 'Game')[:16], font_size=sp(11), size_hint=(0.4, 1))
        top.add_widget(self.name_lbl)
        self.score_lbl = Label(text='0', font_size=sp(14), size_hint=(0.35, 1), color=(1, 1, 0.3, 1))
        top.add_widget(self.score_lbl)
        self.level_lbl = Label(text='Lv1', font_size=sp(11), size_hint=(0.25, 1), color=(0.5, 1, 0.5, 1))
        top.add_widget(self.level_lbl)
        self.layout.add_widget(top)
        
        # HP
        hp = BoxLayout(size_hint=(1, 0.04), pos_hint={'top': 0.92}, padding=(dp(10), 0))
        hp.add_widget(Label(text='HP', font_size=sp(10), size_hint=(0.1, 1), color=(1, 0.5, 0.5, 1)))
        health = self.app.current_game.get('player', {}).get('health', 3)
        self.hp_bar = ProgressBar(max=health, value=health, size_hint=(0.9, 0.6))
        hp.add_widget(self.hp_bar)
        self.layout.add_widget(hp)
        
        # Bottom
        bottom = BoxLayout(size_hint=(1, 0.07), pos_hint={'y': 0}, spacing=dp(5), padding=dp(5))
        with bottom.canvas.before:
            Color(0, 0, 0, 0.5)
            self.bot_bg = Rectangle(pos=bottom.pos, size=bottom.size)
        bottom.bind(pos=lambda *x: setattr(self.bot_bg, 'pos', bottom.pos),
                   size=lambda *x: setattr(self.bot_bg, 'size', bottom.size))
        
        pause = Button(text='||', font_size=sp(14), size_hint=(0.15, 1), background_color=(0.4, 0.4, 0.5, 1), background_normal='')
        pause.bind(on_release=lambda x: setattr(self.engine, 'paused', not self.engine.paused) if self.engine else None)
        bottom.add_widget(pause)
        
        self.combo_lbl = Label(text='', font_size=sp(12), size_hint=(0.5, 1), color=(1, 0.8, 0.2, 1))
        bottom.add_widget(self.combo_lbl)
        
        exit_btn = Button(text='X', font_size=sp(12), size_hint=(0.15, 1), background_color=(0.5, 0.3, 0.3, 1), background_normal='')
        exit_btn.bind(on_release=lambda x: self.exit_game())
        bottom.add_widget(exit_btn)
        
        self.layout.add_widget(bottom)
        
        self.engine.start_game()
        self.update_event = Clock.schedule_interval(self.update, 1/60)
    
    def update(self, dt):
        if not self.engine:
            return False
        
        self.engine.update(dt)
        self.score_lbl.text = str(self.engine.score)
        self.level_lbl.text = f"Lv{self.engine.level}"
        
        if self.engine.player:
            self.hp_bar.value = self.engine.player.health
            self.combo_lbl.text = f"x{self.engine.combo}" if self.engine.combo > 1 else ""
        
        if self.engine.game_over:
            if self.update_event:
                self.update_event.cancel()
            self.app.last_stats = {
                'score': self.engine.score, 'level': self.engine.level,
                'time': self.engine.game_time, 'enemies': self.engine.enemies_killed,
                'combo': self.engine.max_combo, 'name': self.app.current_game.get('name', 'Game')
            }
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'results'), 1)
            return False
        return True
    
    def exit_game(self):
        if self.update_event:
            self.update_event.cancel()
        self.manager.current = 'home'
    
    def on_leave(self):
        if self.update_event:
            self.update_event.cancel()


class ResultsScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(8))
        self.add_widget(self.layout)
    
    def on_pre_enter(self):
        self.layout.clear_widgets()
        s = self.app.last_stats or {}
        
        self.layout.add_widget(Label(text='GAME OVER', font_size=sp(28), size_hint=(1, 0.12), color=(1, 0.5, 0.5, 1), bold=True))
        self.layout.add_widget(Label(text=s.get('name', '')[:20], font_size=sp(13), size_hint=(1, 0.06), color=(0.7, 0.7, 0.9, 1)))
        self.layout.add_widget(Label(text=f"SCORE: {s.get('score', 0)}", font_size=sp(30), size_hint=(1, 0.15), color=(1, 1, 0.3, 1), bold=True))
        
        grid = GridLayout(cols=2, size_hint=(1, 0.25), spacing=dp(5), padding=dp(5))
        for name, val in [('Level', s.get('level', 1)), ('Time', f"{int(s.get('time', 0))}s"),
                          ('Enemies', s.get('enemies', 0)), ('Combo', f"x{s.get('combo', 0)}")]:
            grid.add_widget(Label(text=name, font_size=sp(11), color=(0.6, 0.6, 0.7, 1)))
            grid.add_widget(Label(text=str(val), font_size=sp(13), color=(0.9, 0.9, 1, 1), bold=True))
        self.layout.add_widget(grid)
        
        btns = BoxLayout(size_hint=(1, 0.12), spacing=dp(8))
        retry = Button(text='RETRY', font_size=sp(14), background_color=(0.3, 0.5, 0.3, 1), background_normal='')
        retry.bind(on_release=lambda x: setattr(self.manager, 'current', 'game'))
        btns.add_widget(retry)
        new_btn = Button(text='NEW', font_size=sp(14), background_color=(0.3, 0.3, 0.5, 1), background_normal='')
        new_btn.bind(on_release=lambda x: setattr(self.manager, 'current', 'home'))
        btns.add_widget(new_btn)
        self.layout.add_widget(btns)
        
        home = Button(text='HOME', font_size=sp(12), size_hint=(1, 0.08), background_color=(0.3, 0.3, 0.35, 1), background_normal='')
        home.bind(on_release=lambda x: setattr(self.manager, 'current', 'home'))
        self.layout.add_widget(home)


class HistoryScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        
        layout = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(6))
        layout.add_widget(Label(text='HISTORY', font_size=sp(20), size_hint=(1, 0.08), color=(0.5, 0.8, 1, 1), bold=True))
        
        scroll = ScrollView(size_hint=(1, 0.82))
        self.grid = GridLayout(cols=1, spacing=dp(4), size_hint_y=None, padding=dp(3))
        self.grid.bind(minimum_height=self.grid.setter('height'))
        scroll.add_widget(self.grid)
        layout.add_widget(scroll)
        
        back = Button(text='< BACK', font_size=sp(13), size_hint=(1, 0.07), background_color=(0.3, 0.3, 0.35, 1), background_normal='')
        back.bind(on_release=lambda x: self.go_back())
        layout.add_widget(back)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.refresh()
    
    def refresh(self):
        self.grid.clear_widgets()
        history = self.app.load_history_list()
        
        if not history:
            self.grid.add_widget(Label(text='No games yet!', font_size=sp(13), size_hint_y=None, height=dp(50)))
            return
        
        for g in reversed(history[-12:]):
            tag = "[AI]" if g.get('ai_generated', True) else "[Off]"
            btn = Button(text=f"{tag} {g.get('name', '?')[:16]}", font_size=sp(11), size_hint_y=None, height=dp(45),
                        background_color=(0.15, 0.18, 0.22, 1), background_normal='')
            btn.bind(on_release=lambda x, gg=g: self.play(gg))
            self.grid.add_widget(btn)
    
    def play(self, g):
        self.app.current_game = g
        self.manager.current = 'game'
    
    def go_back(self):
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'home'


# ==================== ПРИЛОЖЕНИЕ ====================

class GameGeneratorApp(App):
    def build(self):
        self.title = 'AI Game Generator'
        self.groq_client = GroqClient()
        self.current_prompt = ""
        self.current_game = None
        self.last_stats = None
        self.data_path = get_data_path()
        
        try:
            os.makedirs(self.data_path, exist_ok=True)
        except:
            pass
        
        key = self.load_api_key()
        if key:
            self.groq_client.set_api_key(key)
        
        model = self.load_model()
        if model:
            self.groq_client.set_model(model)
        
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(HomeScreen(self, name='home'))
        sm.add_widget(SettingsScreen(self, name='settings'))
        sm.add_widget(LoadingScreen(self, name='loading'))
        sm.add_widget(GameScreen(self, name='game'))
        sm.add_widget(ResultsScreen(self, name='results'))
        sm.add_widget(HistoryScreen(self, name='history'))
        
        return sm
    
    def build_config(self, ai_cfg):
        return {
            'name': ai_cfg.get('name', 'AI Game'),
            'description': ai_cfg.get('description', 'AI generated'),
            'theme': ai_cfg.get('theme', 'space'),
            'game_type': ai_cfg.get('game_type', 'shooter'),
            'player': ai_cfg.get('player', {'shape': 'triangle', 'color': [0.3, 0.7, 1], 'speed': 5, 'health': 3}),
            'enemies': ai_cfg.get('enemies', [{'color': [1, 0.3, 0.3], 'speed': 1, 'health': 1, 'points': 10}]),
            'items': ai_cfg.get('items', [{'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1}]),
            'background_colors': ai_cfg.get('background_colors', [[0.1, 0.1, 0.2]]),
            'created_at': time.time(),
            'ai_generated': True
        }
    
    def save_api_key(self, key):
        try:
            with open(os.path.join(self.data_path, 'key.txt'), 'w') as f:
                f.write(key)
        except:
            pass
    
    def load_api_key(self):
        try:
            path = os.path.join(self.data_path, 'key.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        except:
            pass
        return ""
    
    def save_model(self, m):
        try:
            with open(os.path.join(self.data_path, 'model.txt'), 'w') as f:
                f.write(m)
        except:
            pass
    
    def load_model(self):
        try:
            path = os.path.join(self.data_path, 'model.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        except:
            pass
        return ""
    
    def save_history(self, g):
        try:
            history = self.load_history_list()
            history.append(g)
            history = history[-20:]
            with open(os.path.join(self.data_path, 'history.json'), 'w') as f:
                json.dump(history, f)
        except:
            pass
    
    def load_history_list(self):
        try:
            path = os.path.join(self.data_path, 'history.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []


if __name__ == '__main__':
    GameGeneratorApp().run()

"""
AI GAME GENERATOR with Groq API
Генератор мини-игр с использованием настоящего ИИ
"""

import json
import random
import math
import time
import os
import re
import threading
import urllib.request
import urllib.error
from functools import partial

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
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Ellipse, Line, Triangle, RoundedRectangle
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.metrics import dp, sp
from kivy.utils import platform
from kivy.core.window import Window
from kivy.properties import NumericProperty, StringProperty, BooleanProperty


# ==================== GROQ КОНФИГУРАЦИЯ ====================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Модели Groq (рекомендованные вверху)
GROQ_MODELS = [
    # Рекомендованные
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "recommended": True, "speed": "Fast", "quality": "Best"},
    {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B", "recommended": True, "speed": "Fast", "quality": "Excellent"},
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant", "recommended": True, "speed": "Ultra Fast", "quality": "Good"},
    
    # Другие модели
    {"id": "llama3-70b-8192", "name": "Llama 3 70B", "recommended": False, "speed": "Fast", "quality": "Great"},
    {"id": "llama3-8b-8192", "name": "Llama 3 8B", "recommended": False, "speed": "Very Fast", "quality": "Good"},
    {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "recommended": False, "speed": "Fast", "quality": "Great"},
    {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "recommended": False, "speed": "Very Fast", "quality": "Good"},
    {"id": "gemma-7b-it", "name": "Gemma 7B", "recommended": False, "speed": "Very Fast", "quality": "Good"},
    {"id": "llama-3.2-90b-vision-preview", "name": "Llama 3.2 90B Vision", "recommended": False, "speed": "Medium", "quality": "Best"},
    {"id": "llama-3.2-11b-vision-preview", "name": "Llama 3.2 11B Vision", "recommended": False, "speed": "Fast", "quality": "Good"},
]

# Промпт для генерации игры
GAME_GENERATION_PROMPT = """You are a game designer AI. Based on the user's description, generate a unique mini-game configuration.

User's game idea: {prompt}

Generate a JSON response with the following structure (ONLY JSON, no other text):
{{
    "name": "Creative game name",
    "theme": "space/fantasy/zombie/cyber/ocean/nature/candy",
    "game_type": "shooter/runner/collector/survival/avoider",
    "description": "Short game description",
    "player": {{
        "shape": "circle/triangle/square",
        "color": [R, G, B] (values 0-1),
        "speed": 3-8,
        "health": 3-5,
        "special_ability": "shield/dash/magnet/none"
    }},
    "enemies": [
        {{
            "name": "Enemy name",
            "shape": "circle/triangle/square/hexagon",
            "color": [R, G, B],
            "speed": 0.5-3,
            "health": 1-5,
            "points": 10-100,
            "behavior": "chase/patrol/zigzag/spiral"
        }}
    ],
    "items": [
        {{
            "name": "Item name",
            "color": [R, G, B],
            "effect": "heal/shield/speed/magnet/double_points/clear_enemies",
            "value": 1-10
        }}
    ],
    "background_colors": [[R,G,B], [R,G,B], [R,G,B]],
    "difficulty_curve": "easy/normal/hard",
    "special_mechanics": "Description of unique mechanics"
}}

Be creative! Make the game unique based on the user's description. Include 2-4 enemy types and 3-5 item types.
IMPORTANT: Return ONLY valid JSON, no explanations or markdown."""

FUN_FACTS = [
    "Groq's LPU can process 500+ tokens per second!",
    "The first arcade game was Computer Space (1971)",
    "Pac-Man was designed to appeal to women",
    "Mario has appeared in over 200 video games",
    "The Game Boy lasted 10 hours on 4 AA batteries",
    "Tetris was created on an Electronika 60",
    "Sonic was designed to be 'cooler' than Mario",
    "The PlayStation was originally a Nintendo project",
    "Minecraft has sold over 300 million copies",
    "The first Easter egg was in Adventure (1980)",
    "AI is analyzing your creative vision...",
    "Neural networks are designing your game...",
    "Generating unique enemy behaviors...",
    "Crafting personalized power-ups...",
    "Building your custom game world...",
    "Optimizing gameplay mechanics...",
    "Almost ready to play your creation...",
]

LOADING_MESSAGES = [
    "Connecting to Groq AI...",
    "Sending your idea to the cloud...",
    "AI is reading your prompt...",
    "Generating game concept...",
    "Designing characters...",
    "Creating enemies...",
    "Adding power-ups...",
    "Balancing difficulty...",
    "Finalizing game...",
    "Preparing to launch...",
    "Game ready!",
]


# ==================== УТИЛИТЫ ====================

def get_data_path():
    if platform == 'android':
        try:
            from android.storage import app_storage_path
            return app_storage_path()
        except:
            pass
    return os.path.expanduser('~')


def lerp(a, b, t):
    return a + (b - a) * t


# ==================== GROQ API КЛИЕНТ ====================

class GroqClient:
    def __init__(self):
        self.api_key = ""
        self.model = "llama-3.3-70b-versatile"
        self.last_error = None
        self.last_response = None
    
    def set_api_key(self, key):
        self.api_key = key.strip()
    
    def set_model(self, model_id):
        self.model = model_id
    
    def generate_game(self, prompt, callback):
        """Асинхронная генерация игры через Groq API"""
        def api_call():
            try:
                full_prompt = GAME_GENERATION_PROMPT.format(prompt=prompt)
                
                data = json.dumps({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a creative game designer. Always respond with valid JSON only."
                        },
                        {
                            "role": "user", 
                            "content": full_prompt
                        }
                    ],
                    "temperature": 0.8,
                    "max_tokens": 2000
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    GROQ_API_URL,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    content = result['choices'][0]['message']['content']
                    
                    # Извлекаем JSON из ответа
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        game_config = json.loads(json_match.group())
                        self.last_response = game_config
                        Clock.schedule_once(lambda dt: callback(True, game_config), 0)
                    else:
                        Clock.schedule_once(lambda dt: callback(False, "Invalid AI response"), 0)
                        
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8')
                try:
                    error_json = json.loads(error_body)
                    error_msg = error_json.get('error', {}).get('message', str(e))
                except:
                    error_msg = str(e)
                self.last_error = error_msg
                Clock.schedule_once(lambda dt: callback(False, error_msg), 0)
                
            except Exception as e:
                self.last_error = str(e)
                Clock.schedule_once(lambda dt: callback(False, str(e)), 0)
        
        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
    
    def test_connection(self, callback):
        """Тест подключения к API"""
        def test_call():
            try:
                data = json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Say 'OK' if you can hear me."}],
                    "max_tokens": 10
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    GROQ_API_URL,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    Clock.schedule_once(lambda dt: callback(True, "Connection successful!"), 0)
                    
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    Clock.schedule_once(lambda dt: callback(False, "Invalid API Key"), 0)
                else:
                    Clock.schedule_once(lambda dt: callback(False, f"Error: {e.code}"), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: callback(False, str(e)), 0)
        
        thread = threading.Thread(target=test_call)
        thread.daemon = True
        thread.start()


# ==================== ИГРОВЫЕ ОБЪЕКТЫ ====================

class GameObject:
    def __init__(self, x, y, size, color, shape='circle'):
        self.x = x
        self.y = y
        self.size = size
        self.color = tuple(color) if isinstance(color, list) else color
        self.shape = shape
        self.vx = 0
        self.vy = 0
        self.active = True
        self.rotation = 0
    
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.rotation += dt * 50
    
    def collides_with(self, other):
        dist = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        return dist < (self.size + other.size) / 2


class Player(GameObject):
    def __init__(self, x, y, config):
        color = config.get('color', [0.3, 0.7, 1])
        super().__init__(x, y, dp(30), color, config.get('shape', 'circle'))
        self.max_health = config.get('health', 3)
        self.health = self.max_health
        self.speed = config.get('speed', 5)
        self.special_ability = config.get('special_ability', 'none')
        self.shield = 0
        self.magnet = 0
        self.double_points = 0
        self.invincible = 0
        self.dash_cooldown = 0
        self.trail = []
    
    def update(self, dt):
        super().update(dt)
        self.shield = max(0, self.shield - dt)
        self.magnet = max(0, self.magnet - dt)
        self.double_points = max(0, self.double_points - dt)
        self.invincible = max(0, self.invincible - dt)
        self.dash_cooldown = max(0, self.dash_cooldown - dt)
        
        self.trail.append((self.x, self.y, time.time()))
        self.trail = [(x, y, t) for x, y, t in self.trail if time.time() - t < 0.3]
    
    def take_damage(self):
        if self.invincible > 0 or self.shield > 0:
            self.shield = max(0, self.shield - 1)
            return False
        self.health -= 1
        self.invincible = 1.5
        return self.health <= 0
    
    def use_ability(self):
        if self.special_ability == 'dash' and self.dash_cooldown <= 0:
            self.dash_cooldown = 3
            return 'dash'
        elif self.special_ability == 'shield' and self.dash_cooldown <= 0:
            self.shield = 3
            self.dash_cooldown = 5
            return 'shield'
        return None


class Enemy(GameObject):
    def __init__(self, x, y, config):
        color = config.get('color', [1, 0.3, 0.3])
        size = dp(25) * config.get('size', 1)
        super().__init__(x, y, size, color, config.get('shape', 'circle'))
        self.speed = config.get('speed', 1)
        self.health = config.get('health', 1)
        self.max_health = self.health
        self.points = config.get('points', 10)
        self.name = config.get('name', 'Enemy')
        self.behavior = config.get('behavior', random.choice(['chase', 'patrol', 'zigzag', 'spiral']))
        self.timer = 0
    
    def update(self, dt, target_x=None, target_y=None):
        self.timer += dt
        
        if self.behavior == 'chase' and target_x is not None:
            dx = target_x - self.x
            dy = target_y - self.y
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
        elif self.behavior == 'spiral':
            self.vx = math.cos(self.timer * 3) * self.speed * dp(50)
            self.vy = math.sin(self.timer * 3) * self.speed * dp(50) - dp(30)
        
        super().update(dt)
    
    def take_damage(self, amount=1):
        self.health -= amount
        return self.health <= 0


class Item(GameObject):
    def __init__(self, x, y, config):
        color = config.get('color', [1, 1, 0.3])
        super().__init__(x, y, dp(20), color, 'circle')
        self.item_type = config.get('type', 'heal')
        self.effect = config.get('effect', 'heal')
        self.value = config.get('value', 1)
        self.name = config.get('name', 'Item')
        self.pulse = 0
    
    def update(self, dt):
        self.pulse += dt * 5
        self.vy = -dp(30)
        super().update(dt)


class Bullet(GameObject):
    def __init__(self, x, y, vx, vy, color):
        super().__init__(x, y, dp(8), color, 'circle')
        self.vx = vx
        self.vy = vy
        self.damage = 1


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = tuple(color) if isinstance(color, list) else color
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
    
    def __init__(self, game_config, **kwargs):
        super().__init__(**kwargs)
        self.config = game_config
        self.player = None
        self.enemies = []
        self.items = []
        self.bullets = []
        self.particles = []
        
        self.game_time = 0
        self.spawn_timer = 0
        self.level_timer = 0
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
        self.setup_background()
        
        self.bind(size=self.on_size_change)
    
    def on_size_change(self, *args):
        self.setup_background()
        if self.player:
            self.player.x = max(self.player.size/2, 
                               min(self.width - self.player.size/2, self.player.x))
            self.player.y = max(self.player.size/2, 
                               min(self.height - self.player.size/2, self.player.y))
    
    def setup_background(self):
        self.stars = []
        for _ in range(60):
            self.stars.append({
                'x': random.uniform(0, max(100, self.width)),
                'y': random.uniform(0, max(100, self.height)),
                'size': random.uniform(1, 3),
                'speed': random.uniform(20, 80),
                'brightness': random.uniform(0.3, 1.0)
            })
    
    def start_game(self):
        player_config = self.config.get('player', {})
        self.player = Player(
            self.width / 2,
            self.height * 0.2,
            player_config
        )
        
        self.enemies = []
        self.items = []
        self.bullets = []
        self.particles = []
        
        self.score = 0
        self.level = 1
        self.game_time = 0
        self.spawn_timer = 0
        self.level_timer = 0
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
        self.level_timer += dt
        self.combo_timer -= dt
        
        if self.combo_timer <= 0:
            self.combo = 0
        
        # Уровни
        if self.enemies_killed >= 5 + self.level * 3 and not self.enemies:
            self.level_up()
        
        # Управление
        if self.touching:
            target_x, target_y = self.touch_pos
            dx = target_x - self.player.x
            dy = target_y - self.player.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 5:
                speed = self.player.speed * dp(60) * dt
                self.player.x += (dx / dist) * min(speed, dist)
                self.player.y += (dy / dist) * min(speed, dist)
        
        # Границы
        self.player.x = max(self.player.size/2, 
                           min(self.width - self.player.size/2, self.player.x))
        self.player.y = max(self.player.size/2, 
                           min(self.height - self.player.size/2, self.player.y))
        
        self.player.update(dt)
        
        # Автострельба
        game_type = self.config.get('game_type', 'collector')
        if game_type == 'shooter' and int(self.game_time * 5) % 2 == 0:
            if len(self.bullets) < 10:
                self.bullets.append(Bullet(
                    self.player.x, self.player.y + self.player.size/2,
                    0, dp(400),
                    self.player.color
                ))
        
        # Спавн врагов
        difficulty = self.config.get('difficulty_curve', 'normal')
        spawn_rate = {'easy': 2.5, 'normal': 2.0, 'hard': 1.5}.get(difficulty, 2.0)
        spawn_rate /= (1 + self.level * 0.1)
        
        if self.spawn_timer > spawn_rate and len(self.enemies) < 15:
            self.spawn_enemy()
            self.spawn_timer = 0
        
        # Спавн предметов
        if random.random() < 0.005:
            self.spawn_item()
        
        # Враги
        for enemy in self.enemies[:]:
            enemy.update(dt, self.player.x, self.player.y)
            
            if enemy.y < -enemy.size or enemy.y > self.height + enemy.size:
                self.enemies.remove(enemy)
                continue
            
            if enemy.collides_with(self.player):
                if self.player.take_damage():
                    self.game_over = True
                else:
                    self.spawn_particles(enemy.x, enemy.y, enemy.color, 10)
                    self.enemies.remove(enemy)
        
        # Предметы
        for item in self.items[:]:
            if self.player.magnet > 0:
                dx = self.player.x - item.x
                dy = self.player.y - item.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0 and dist < dp(150):
                    item.x += (dx / dist) * dp(200) * dt
                    item.y += (dy / dist) * dp(200) * dt
            
            item.update(dt)
            
            if item.y < -item.size:
                self.items.remove(item)
                continue
            
            if item.collides_with(self.player):
                self.collect_item(item)
                self.items.remove(item)
        
        # Пули
        for bullet in self.bullets[:]:
            bullet.update(dt)
            
            if bullet.y > self.height + bullet.size or bullet.y < -bullet.size:
                self.bullets.remove(bullet)
                continue
            
            for enemy in self.enemies[:]:
                if bullet.collides_with(enemy):
                    if enemy.take_damage(bullet.damage):
                        self.kill_enemy(enemy)
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    self.spawn_particles(bullet.x, bullet.y, bullet.color, 5)
                    break
        
        # Частицы
        self.particles = [p for p in self.particles if p.update(dt)]
        
        # Звёзды
        for star in self.stars:
            star['y'] -= star['speed'] * dt
            if star['y'] < 0:
                star['y'] = self.height
                star['x'] = random.uniform(0, self.width)
        
        self.draw()
    
    def level_up(self):
        self.level += 1
        self.enemies_killed = 0
        self.level_timer = 0
        self.score += self.level * 100
        
        for _ in range(30):
            self.particles.append(Particle(
                random.uniform(0, self.width),
                random.uniform(0, self.height),
                (1, 1, 0.3)
            ))
    
    def spawn_enemy(self):
        enemies_config = self.config.get('enemies', [])
        if not enemies_config:
            enemies_config = [{'name': 'Enemy', 'color': [1, 0.3, 0.3], 'speed': 1, 'health': 1, 'points': 10}]
        
        enemy_config = random.choice(enemies_config).copy()
        enemy_config['speed'] *= (1 + self.level * 0.1)
        
        x = random.uniform(dp(30), self.width - dp(30))
        y = self.height + dp(30)
        
        self.enemies.append(Enemy(x, y, enemy_config))
    
    def spawn_item(self):
        items_config = self.config.get('items', [])
        if not items_config:
            items_config = [{'name': 'Health', 'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1}]
        
        item_config = random.choice(items_config).copy()
        x = random.uniform(dp(30), self.width - dp(30))
        y = self.height + dp(20)
        self.items.append(Item(x, y, item_config))
    
    def kill_enemy(self, enemy):
        points = enemy.points
        if self.player.double_points > 0:
            points *= 2
        
        self.combo += 1
        self.combo_timer = 2.0
        self.max_combo = max(self.max_combo, self.combo)
        
        points *= (1 + self.combo * 0.1)
        self.score += int(points)
        
        self.enemies_killed += 1
        self.spawn_particles(enemy.x, enemy.y, enemy.color, 15)
        
        if enemy in self.enemies:
            self.enemies.remove(enemy)
    
    def collect_item(self, item):
        self.items_collected += 1
        self.score += 25
        
        effect = item.effect
        value = item.value
        
        if effect == 'heal':
            self.player.health = min(self.player.max_health, self.player.health + value)
        elif effect == 'shield':
            self.player.shield += value
        elif effect == 'speed':
            self.player.speed *= 1.3
        elif effect == 'magnet':
            self.player.magnet += value
        elif effect == 'double_points':
            self.player.double_points += value
        elif effect == 'clear_enemies':
            for enemy in self.enemies[:]:
                self.kill_enemy(enemy)
        
        self.spawn_particles(item.x, item.y, item.color, 10)
    
    def spawn_particles(self, x, y, color, count):
        for _ in range(count):
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
        
        bg_colors = self.config.get('background_colors', [[0.1, 0.1, 0.2]])
        if not bg_colors:
            bg_colors = [[0.1, 0.1, 0.2]]
        
        with self.canvas:
            # Фон
            bg = bg_colors[0]
            Color(bg[0], bg[1], bg[2], 1)
            Rectangle(pos=self.pos, size=self.size)
            
            # Градиент
            if len(bg_colors) > 1:
                bg2 = bg_colors[1]
                Color(bg2[0], bg2[1], bg2[2], 0.5)
                Rectangle(pos=(self.x, self.y + self.height * 0.5), 
                         size=(self.width, self.height * 0.5))
            
            # Звёзды
            for star in self.stars:
                b = star['brightness']
                Color(b, b, b * 1.1, b)
                Ellipse(pos=(star['x'], star['y']), 
                       size=(star['size'], star['size']))
            
            # Частицы
            for p in self.particles:
                c = p.color
                Color(c[0], c[1], c[2], p.life)
                Ellipse(pos=(p.x - p.size/2, p.y - p.size/2),
                       size=(p.size, p.size))
            
            # Предметы
            for item in self.items:
                pulse = 1 + 0.2 * math.sin(item.pulse)
                size = item.size * pulse
                c = item.color
                Color(c[0], c[1], c[2], 0.3)
                Ellipse(pos=(item.x - size, item.y - size),
                       size=(size * 2, size * 2))
                Color(c[0], c[1], c[2], 1)
                Ellipse(pos=(item.x - item.size/2, item.y - item.size/2),
                       size=(item.size, item.size))
            
            # Пули
            for bullet in self.bullets:
                c = bullet.color
                Color(c[0], c[1], c[2], 1)
                Ellipse(pos=(bullet.x - bullet.size/2, bullet.y - bullet.size/2),
                       size=(bullet.size, bullet.size))
            
            # Враги
            for enemy in self.enemies:
                c = enemy.color
                Color(0, 0, 0, 0.3)
                self.draw_shape(enemy.shape, enemy.x + 3, enemy.y - 3, enemy.size)
                Color(c[0], c[1], c[2], 1)
                self.draw_shape(enemy.shape, enemy.x, enemy.y, enemy.size)
                
                if enemy.health < enemy.max_health:
                    bar_width = enemy.size
                    bar_height = dp(4)
                    Color(0.3, 0.3, 0.3, 0.8)
                    Rectangle(pos=(enemy.x - bar_width/2, enemy.y + enemy.size/2 + 5),
                             size=(bar_width, bar_height))
                    Color(0.2, 0.8, 0.2, 1)
                    Rectangle(pos=(enemy.x - bar_width/2, enemy.y + enemy.size/2 + 5),
                             size=(bar_width * (enemy.health / enemy.max_health), bar_height))
            
            # Игрок
            if self.player:
                # Trail
                for i, (tx, ty, _) in enumerate(self.player.trail):
                    alpha = i / len(self.player.trail) * 0.3
                    c = self.player.color
                    Color(c[0], c[1], c[2], alpha)
                    size = self.player.size * (0.3 + 0.7 * i / max(1, len(self.player.trail)))
                    self.draw_shape(self.player.shape, tx, ty, size)
                
                # Свечение
                c = self.player.color
                glow_size = self.player.size * 1.5
                Color(c[0], c[1], c[2], 0.2)
                Ellipse(pos=(self.player.x - glow_size/2, self.player.y - glow_size/2),
                       size=(glow_size, glow_size))
                
                # Игрок
                if self.player.invincible > 0 and int(self.player.invincible * 10) % 2:
                    Color(c[0], c[1], c[2], 0.5)
                else:
                    Color(c[0], c[1], c[2], 1)
                self.draw_shape(self.player.shape, self.player.x, self.player.y, self.player.size)
                
                # Щит
                if self.player.shield > 0:
                    Color(0.3, 0.7, 1, 0.4)
                    Line(circle=(self.player.x, self.player.y, self.player.size * 0.8), width=2)
    
    def draw_shape(self, shape, x, y, size):
        half = size / 2
        if shape == 'circle':
            Ellipse(pos=(x - half, y - half), size=(size, size))
        elif shape == 'square' or shape == 'rectangle':
            Rectangle(pos=(x - half, y - half), size=(size, size))
        elif shape == 'triangle':
            Triangle(points=[
                x, y + half,
                x - half, y - half,
                x + half, y - half
            ])
        elif shape == 'hexagon':
            Ellipse(pos=(x - half * 0.9, y - half * 0.9), size=(size * 0.9, size * 0.9))


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


# ==================== ЭКРАН НАСТРОЙКИ API ====================

class SettingsScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.selected_model = GROQ_MODELS[0]['id']
        self.filtered_models = GROQ_MODELS.copy()
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(8))
        
        # Заголовок
        header = BoxLayout(size_hint=(1, 0.08))
        
        back_btn = Button(
            text='<',
            size_hint=(0.15, 1),
            font_size=sp(24),
            background_color=(0.3, 0.3, 0.4, 1),
            background_normal=''
        )
        back_btn.bind(on_release=lambda x: self.go_back())
        header.add_widget(back_btn)
        
        header.add_widget(Label(
            text='API Settings',
            font_size=sp(22),
            color=(0.8, 0.8, 1, 1),
            bold=True
        ))
        
        layout.add_widget(header)
        
        # Красивый блок для API ключа
        api_box = BoxLayout(orientation='vertical', size_hint=(1, 0.25), spacing=dp(5))
        
        # Заголовок блока
        api_header = BoxLayout(size_hint=(1, 0.25))
        api_header.add_widget(Label(
            text='GROQ API Key',
            font_size=sp(16),
            color=(0.5, 0.8, 1, 1),
            halign='left',
            bold=True
        ))
        
        self.status_indicator = Label(
            text='',
            font_size=sp(12),
            size_hint=(0.4, 1),
            color=(0.5, 0.5, 0.5, 1)
        )
        api_header.add_widget(self.status_indicator)
        api_box.add_widget(api_header)
        
        # Поле ввода ключа с красивым фоном
        key_container = FloatLayout(size_hint=(1, 0.45))
        
        with key_container.canvas.before:
            Color(0.12, 0.12, 0.18, 1)
            self.key_bg = RoundedRectangle(
                pos=key_container.pos,
                size=key_container.size,
                radius=[dp(10)]
            )
        key_container.bind(
            pos=lambda *x: setattr(self.key_bg, 'pos', key_container.pos),
            size=lambda *x: setattr(self.key_bg, 'size', key_container.size)
        )
        
        self.api_key_input = TextInput(
            hint_text='gsk_xxxxxxxxxxxxxxxxx...',
            multiline=False,
            password=True,
            font_size=sp(14),
            size_hint=(0.98, 0.9),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            background_color=(0, 0, 0, 0),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(0.5, 0.8, 1, 1),
            hint_text_color=(0.4, 0.4, 0.5, 1),
            padding=[dp(15), dp(10)]
        )
        
        # Загружаем сохранённый ключ
        saved_key = self.app.load_api_key()
        if saved_key:
            self.api_key_input.text = saved_key
        
        key_container.add_widget(self.api_key_input)
        api_box.add_widget(key_container)
        
        # Кнопки для ключа
        key_buttons = BoxLayout(size_hint=(1, 0.3), spacing=dp(10))
        
        show_btn = Button(
            text='Show/Hide',
            font_size=sp(12),
            background_color=(0.3, 0.3, 0.4, 1),
            background_normal=''
        )
        show_btn.bind(on_release=lambda x: self.toggle_password())
        key_buttons.add_widget(show_btn)
        
        test_btn = Button(
            text='Test Connection',
            font_size=sp(12),
            background_color=(0.3, 0.5, 0.6, 1),
            background_normal=''
        )
        test_btn.bind(on_release=lambda x: self.test_connection())
        key_buttons.add_widget(test_btn)
        
        save_btn = Button(
            text='Save Key',
            font_size=sp(12),
            background_color=(0.3, 0.5, 0.3, 1),
            background_normal=''
        )
        save_btn.bind(on_release=lambda x: self.save_key())
        key_buttons.add_widget(save_btn)
        
        api_box.add_widget(key_buttons)
        layout.add_widget(api_box)
        
        # Получить ключ
        get_key_btn = Button(
            text='Get FREE API Key at console.groq.com',
            font_size=sp(12),
            size_hint=(1, 0.04),
            background_color=(0.2, 0.4, 0.5, 1),
            background_normal=''
        )
        layout.add_widget(get_key_btn)
        
        # Выбор модели
        layout.add_widget(Label(
            text='Select AI Model:',
            font_size=sp(16),
            size_hint=(1, 0.04),
            color=(0.5, 0.8, 1, 1),
            halign='left',
            bold=True
        ))
        
        # Поиск моделей
        search_box = BoxLayout(size_hint=(1, 0.06), spacing=dp(5))
        
        self.model_search = TextInput(
            hint_text='Search models...',
            multiline=False,
            font_size=sp(14),
            size_hint=(1, 1),
            background_color=(0.12, 0.12, 0.18, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(0.5, 0.8, 1, 1)
        )
        self.model_search.bind(text=self.filter_models)
        search_box.add_widget(self.model_search)
        
        layout.add_widget(search_box)
        
        # Список моделей
        models_scroll = ScrollView(size_hint=(1, 0.38))
        self.models_grid = GridLayout(cols=1, spacing=dp(5), size_hint_y=None, padding=dp(5))
        self.models_grid.bind(minimum_height=self.models_grid.setter('height'))
        
        self.refresh_models_list()
        
        models_scroll.add_widget(self.models_grid)
        layout.add_widget(models_scroll)
        
        # Выбранная модель
        self.selected_label = Label(
            text=f'Selected: {GROQ_MODELS[0]["name"]}',
            font_size=sp(14),
            size_hint=(1, 0.05),
            color=(0.5, 1, 0.5, 1)
        )
        layout.add_widget(self.selected_label)
        
        self.add_widget(layout)
    
    def refresh_models_list(self):
        self.models_grid.clear_widgets()
        
        # Сначала рекомендованные
        for model in self.filtered_models:
            if model['recommended']:
                self.add_model_button(model, True)
        
        # Затем остальные
        for model in self.filtered_models:
            if not model['recommended']:
                self.add_model_button(model, False)
    
    def add_model_button(self, model, is_recommended):
        is_selected = model['id'] == self.selected_model
        
        if is_recommended:
            bg_color = (0.25, 0.35, 0.25, 1) if is_selected else (0.2, 0.3, 0.2, 1)
            prefix = "★ "
        else:
            bg_color = (0.25, 0.25, 0.35, 1) if is_selected else (0.15, 0.15, 0.2, 1)
            prefix = "  "
        
        btn = Button(
            text=f"{prefix}{model['name']}\n   {model['speed']} | {model['quality']}",
            font_size=sp(12),
            size_hint_y=None,
            height=dp(55),
            background_color=bg_color,
            background_normal='',
            halign='left'
        )
        btn.bind(on_release=lambda x, m=model: self.select_model(m))
        self.models_grid.add_widget(btn)
    
    def filter_models(self, instance, text):
        text = text.lower()
        if text:
            self.filtered_models = [m for m in GROQ_MODELS if text in m['name'].lower() or text in m['id'].lower()]
        else:
            self.filtered_models = GROQ_MODELS.copy()
        self.refresh_models_list()
    
    def select_model(self, model):
        self.selected_model = model['id']
        self.selected_label.text = f"Selected: {model['name']}"
        self.app.groq_client.set_model(model['id'])
        self.app.save_selected_model(model['id'])
        self.refresh_models_list()
    
    def toggle_password(self):
        self.api_key_input.password = not self.api_key_input.password
    
    def save_key(self):
        key = self.api_key_input.text.strip()
        self.app.save_api_key(key)
        self.app.groq_client.set_api_key(key)
        self.status_indicator.text = 'Saved!'
        self.status_indicator.color = (0.5, 1, 0.5, 1)
    
    def test_connection(self):
        key = self.api_key_input.text.strip()
        if not key:
            self.status_indicator.text = 'Enter API key first'
            self.status_indicator.color = (1, 0.5, 0.5, 1)
            return
        
        self.app.groq_client.set_api_key(key)
        self.status_indicator.text = 'Testing...'
        self.status_indicator.color = (1, 1, 0.5, 1)
        
        self.app.groq_client.test_connection(self.on_test_result)
    
    def on_test_result(self, success, message):
        if success:
            self.status_indicator.text = 'Connected!'
            self.status_indicator.color = (0.5, 1, 0.5, 1)
        else:
            self.status_indicator.text = message[:20]
            self.status_indicator.color = (1, 0.5, 0.5, 1)
    
    def go_back(self):
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'home'


# ==================== ГЛАВНЫЙ ЭКРАН ====================

class HomeScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
        
        # Заголовок
        layout.add_widget(Label(
            text='AI GAME',
            font_size=sp(42),
            size_hint=(1, 0.1),
            bold=True,
            color=(0.5, 0.8, 1, 1)
        ))
        
        layout.add_widget(Label(
            text='GENERATOR',
            font_size=sp(36),
            size_hint=(1, 0.08),
            bold=True,
            color=(1, 0.5, 0.8, 1)
        ))
        
        layout.add_widget(Label(
            text='Powered by Groq AI',
            font_size=sp(12),
            size_hint=(1, 0.03),
            color=(0.5, 0.5, 0.6, 1)
        ))
        
        # Статус API
        self.api_status = Label(
            text='API: Not configured',
            font_size=sp(12),
            size_hint=(1, 0.04),
            color=(1, 0.5, 0.5, 1)
        )
        layout.add_widget(self.api_status)
        
        # Поле ввода
        layout.add_widget(Label(
            text='Describe your dream game:',
            font_size=sp(14),
            size_hint=(1, 0.04),
            halign='left',
            color=(0.8, 0.8, 0.9, 1)
        ))
        
        self.prompt_input = TextInput(
            hint_text='Example: A space shooter where I pilot a neon ship and fight alien robots',
            multiline=True,
            font_size=sp(14),
            size_hint=(1, 0.18),
            background_color=(0.1, 0.1, 0.15, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(0.5, 0.8, 1, 1),
            hint_text_color=(0.4, 0.4, 0.5, 1)
        )
        layout.add_widget(self.prompt_input)
        
        # Примеры
        layout.add_widget(Label(
            text='Quick ideas:',
            font_size=sp(12),
            size_hint=(1, 0.03),
            color=(0.6, 0.6, 0.7, 1)
        ))
        
        examples_grid = GridLayout(cols=2, size_hint=(1, 0.1), spacing=dp(5))
        examples = [
            ('Космический шутер', 'Space shooter with neon effects and alien enemies'),
            ('Зомби выживание', 'Zombie survival game in a dark apocalyptic world'),
            ('Сладкий мир', 'Cute candy collector in a magical sweet kingdom'),
            ('Киберпанк', 'Cyberpunk robot battle with hacking mechanics'),
        ]
        
        for name, prompt in examples:
            btn = Button(
                text=name,
                font_size=sp(11),
                background_color=(0.2, 0.2, 0.3, 1),
                background_normal=''
            )
            btn.bind(on_release=lambda x, p=prompt: self.set_prompt(p))
            examples_grid.add_widget(btn)
        
        layout.add_widget(examples_grid)
        
        # Кнопка генерации
        self.generate_btn = Button(
            text='GENERATE WITH AI',
            font_size=sp(20),
            size_hint=(1, 0.1),
            background_color=(0.3, 0.6, 0.3, 1),
            background_normal='',
            bold=True
        )
        self.generate_btn.bind(on_release=lambda x: self.generate())
        layout.add_widget(self.generate_btn)
        
        # Нижние кнопки
        bottom_buttons = BoxLayout(size_hint=(1, 0.07), spacing=dp(10))
        
        settings_btn = Button(
            text='API Settings',
            font_size=sp(14),
            background_color=(0.4, 0.3, 0.5, 1),
            background_normal=''
        )
        settings_btn.bind(on_release=lambda x: self.go_settings())
        bottom_buttons.add_widget(settings_btn)
        
        history_btn = Button(
            text='My Games',
            font_size=sp(14),
            background_color=(0.25, 0.25, 0.35, 1),
            background_normal=''
        )
        history_btn.bind(on_release=lambda x: self.go_history())
        bottom_buttons.add_widget(history_btn)
        
        layout.add_widget(bottom_buttons)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.update_api_status()
    
    def update_api_status(self):
        key = self.app.load_api_key()
        if key:
            self.api_status.text = f'API: Connected | Model: {self.app.groq_client.model}'
            self.api_status.color = (0.5, 1, 0.5, 1)
            self.generate_btn.background_color = (0.3, 0.6, 0.3, 1)
        else:
            self.api_status.text = 'API: Not configured - Go to Settings'
            self.api_status.color = (1, 0.5, 0.5, 1)
            self.generate_btn.background_color = (0.4, 0.4, 0.4, 1)
    
    def set_prompt(self, prompt):
        self.prompt_input.text = prompt
    
    def generate(self):
        key = self.app.load_api_key()
        if not key:
            self.api_status.text = 'Please configure API key first!'
            self.api_status.color = (1, 0.5, 0.5, 1)
            return
        
        prompt = self.prompt_input.text.strip()
        if not prompt:
            prompt = "A fun arcade game with colorful graphics"
        
        self.app.current_prompt = prompt
        self.manager.current = 'loading'
    
    def go_settings(self):
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'settings'
    
    def go_history(self):
        self.manager.transition = SlideTransition(direction='left')
        self.manager.current = 'history'


# ==================== ЭКРАН ЗАГРУЗКИ ====================

class LoadingScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.progress = 0
        self.current_message = 0
        self.fact_index = 0
        self.dots_anim = 0
        self.generation_complete = False
        self.generation_error = None
        
        self.layout = FloatLayout()
        
        # AI иконка
        self.ai_label = Label(
            text='AI',
            font_size=sp(48),
            pos_hint={'center_x': 0.5, 'center_y': 0.78},
            color=(0.5, 0.8, 1, 1),
            bold=True
        )
        self.layout.add_widget(self.ai_label)
        
        # Модель
        self.model_label = Label(
            text='',
            font_size=sp(12),
            pos_hint={'center_x': 0.5, 'center_y': 0.72},
            color=(0.5, 0.5, 0.6, 1)
        )
        self.layout.add_widget(self.model_label)
        
        # Прогресс бар
        self.progress_bar = ProgressBar(
            max=100,
            value=0,
            size_hint=(0.8, 0.03),
            pos_hint={'center_x': 0.5, 'center_y': 0.55}
        )
        self.layout.add_widget(self.progress_bar)
        
        # Процент
        self.percent_label = Label(
            text='0%',
            font_size=sp(18),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            color=(0.5, 0.8, 1, 1)
        )
        self.layout.add_widget(self.percent_label)
        
        # Статус
        self.status_label = Label(
            text='Initializing...',
            font_size=sp(14),
            pos_hint={'center_x': 0.5, 'center_y': 0.43},
            color=(0.6, 0.6, 0.7, 1)
        )
        self.layout.add_widget(self.status_label)
        
        # Факт
        self.fact_label = Label(
            text='',
            font_size=sp(13),
            pos_hint={'center_x': 0.5, 'center_y': 0.25},
            color=(0.7, 0.7, 0.5, 1),
            halign='center',
            size_hint=(0.9, 0.15)
        )
        self.fact_label.bind(size=lambda *x: setattr(self.fact_label, 'text_size', (self.fact_label.width, None)))
        self.layout.add_widget(self.fact_label)
        
        # Ошибка
        self.error_label = Label(
            text='',
            font_size=sp(12),
            pos_hint={'center_x': 0.5, 'center_y': 0.15},
            color=(1, 0.5, 0.5, 1),
            halign='center',
            size_hint=(0.9, 0.1)
        )
        self.error_label.bind(size=lambda *x: setattr(self.error_label, 'text_size', (self.error_label.width, None)))
        self.layout.add_widget(self.error_label)
        
        self.add_widget(self.layout)
    
    def on_pre_enter(self):
        self.progress = 0
        self.generation_complete = False
        self.generation_error = None
        self.error_label.text = ''
        self.model_label.text = f'Using: {self.app.groq_client.model}'
        self.start_generation()
    
    def start_generation(self):
        # Запускаем анимацию
        Clock.schedule_interval(self.update_loading, 0.05)
        
        # Запускаем генерацию через API
        self.app.groq_client.generate_game(
            self.app.current_prompt,
            self.on_generation_result
        )
    
    def on_generation_result(self, success, result):
        if success:
            self.generation_complete = True
            self.app.ai_generated_config = result
        else:
            self.generation_error = result
    
    def update_loading(self, dt):
        self.dots_anim += dt
        
        # Проверяем ошибку
        if self.generation_error:
            Clock.unschedule(self.update_loading)
            self.error_label.text = f"Error: {self.generation_error}"
            self.status_label.text = "Generation failed"
            self.percent_label.text = "!"
            
            # Возврат через 3 секунды
            Clock.schedule_once(lambda dt: self.go_back(), 3)
            return False
        
        # Проверяем завершение
        if self.generation_complete:
            self.progress = 100
            self.progress_bar.value = 100
            self.percent_label.text = "100%"
            self.status_label.text = "Game Ready!"
            
            Clock.unschedule(self.update_loading)
            Clock.schedule_once(lambda dt: self.go_to_game(), 0.5)
            return False
        
        # Анимация прогресса (до 90%)
        if self.progress < 90:
            self.progress += random.uniform(0.3, 1.5)
        
        self.progress_bar.value = self.progress
        self.percent_label.text = f"{int(self.progress)}%"
        
        # Статус
        msg_index = min(int(self.progress / 10), len(LOADING_MESSAGES) - 1)
        dots = '.' * (int(self.dots_anim * 2) % 4)
        self.status_label.text = LOADING_MESSAGES[msg_index] + dots
        
        # AI анимация
        pulse = 1 + 0.1 * math.sin(self.dots_anim * 5)
        self.ai_label.font_size = sp(48 * pulse)
        
        # Факт
        if int(self.dots_anim) % 4 == 0 and int(self.dots_anim) != self.fact_index:
            self.fact_index = int(self.dots_anim)
            self.fact_label.text = f"Did you know?\n{random.choice(FUN_FACTS)}"
        
        return True
    
    def go_to_game(self):
        # Создаём конфиг игры из ответа AI
        self.app.current_game = self.app.build_game_config(self.app.ai_generated_config)
        self.app.save_game_to_history(self.app.current_game)
        self.manager.current = 'game'
    
    def go_back(self):
        self.manager.current = 'home'


# ==================== ИГРОВОЙ ЭКРАН ====================

class GameScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.engine = None
        
        self.layout = FloatLayout()
        self.add_widget(self.layout)
    
    def on_pre_enter(self):
        self.setup_game()
    
    def setup_game(self):
        self.layout.clear_widgets()
        
        if not self.app.current_game:
            return
        
        # Движок
        self.engine = GameEngine(self.app.current_game, size_hint=(1, 0.85))
        self.layout.add_widget(self.engine)
        
        # Верхняя панель
        top_panel = BoxLayout(
            size_hint=(1, 0.08),
            pos_hint={'top': 1},
            padding=dp(5),
            spacing=dp(5)
        )
        
        with top_panel.canvas.before:
            Color(0, 0, 0, 0.5)
            self.top_bg = Rectangle(pos=top_panel.pos, size=top_panel.size)
        top_panel.bind(pos=lambda *x: setattr(self.top_bg, 'pos', top_panel.pos),
                      size=lambda *x: setattr(self.top_bg, 'size', top_panel.size))
        
        self.game_name_label = Label(
            text=self.app.current_game.get('name', 'Game'),
            font_size=sp(13),
            size_hint=(0.4, 1),
            halign='left',
            color=(0.8, 0.8, 1, 1)
        )
        top_panel.add_widget(self.game_name_label)
        
        self.score_label = Label(
            text='Score: 0',
            font_size=sp(16),
            size_hint=(0.35, 1),
            color=(1, 1, 0.3, 1)
        )
        top_panel.add_widget(self.score_label)
        
        self.level_label = Label(
            text='Lv.1',
            font_size=sp(14),
            size_hint=(0.25, 1),
            color=(0.5, 1, 0.5, 1)
        )
        top_panel.add_widget(self.level_label)
        
        self.layout.add_widget(top_panel)
        
        # Здоровье
        health_panel = BoxLayout(
            size_hint=(1, 0.04),
            pos_hint={'top': 0.92},
            padding=(dp(10), 0)
        )
        
        self.health_label = Label(
            text='HP:',
            font_size=sp(12),
            size_hint=(0.15, 1),
            color=(1, 0.5, 0.5, 1)
        )
        health_panel.add_widget(self.health_label)
        
        player_health = self.app.current_game.get('player', {}).get('health', 3)
        self.health_bar = ProgressBar(
            max=player_health,
            value=player_health,
            size_hint=(0.85, 0.6)
        )
        health_panel.add_widget(self.health_bar)
        
        self.layout.add_widget(health_panel)
        
        # Нижняя панель
        bottom_panel = BoxLayout(
            size_hint=(1, 0.07),
            pos_hint={'y': 0},
            spacing=dp(5),
            padding=dp(5)
        )
        
        with bottom_panel.canvas.before:
            Color(0, 0, 0, 0.5)
            self.bottom_bg = Rectangle(pos=bottom_panel.pos, size=bottom_panel.size)
        bottom_panel.bind(pos=lambda *x: setattr(self.bottom_bg, 'pos', bottom_panel.pos),
                         size=lambda *x: setattr(self.bottom_bg, 'size', bottom_panel.size))
        
        pause_btn = Button(
            text='||',
            font_size=sp(18),
            size_hint=(0.15, 1),
            background_color=(0.4, 0.4, 0.5, 1),
            background_normal=''
        )
        pause_btn.bind(on_release=lambda x: self.toggle_pause())
        bottom_panel.add_widget(pause_btn)
        
        self.combo_label = Label(
            text='',
            font_size=sp(14),
            size_hint=(0.4, 1),
            color=(1, 0.8, 0.2, 1)
        )
        bottom_panel.add_widget(self.combo_label)
        
        self.time_label = Label(
            text='0:00',
            font_size=sp(14),
            size_hint=(0.25, 1),
            color=(0.7, 0.7, 0.8, 1)
        )
        bottom_panel.add_widget(self.time_label)
        
        exit_btn = Button(
            text='X',
            font_size=sp(16),
            size_hint=(0.15, 1),
            background_color=(0.5, 0.3, 0.3, 1),
            background_normal=''
        )
        exit_btn.bind(on_release=lambda x: self.confirm_exit())
        bottom_panel.add_widget(exit_btn)
        
        self.layout.add_widget(bottom_panel)
        
        self.engine.start_game()
        Clock.schedule_interval(self.update, 1/60)
    
    def update(self, dt):
        if not self.engine:
            return False
        
        self.engine.update(dt)
        
        self.score_label.text = f"Score: {self.engine.score}"
        self.level_label.text = f"Lv.{self.engine.level}"
        
        if self.engine.player:
            self.health_bar.value = self.engine.player.health
            
            if self.engine.combo > 1:
                self.combo_label.text = f"Combo x{self.engine.combo}!"
            else:
                self.combo_label.text = ""
        
        mins = int(self.engine.game_time) // 60
        secs = int(self.engine.game_time) % 60
        self.time_label.text = f"{mins}:{secs:02d}"
        
        if self.engine.game_over:
            Clock.unschedule(self.update)
            self.app.last_game_stats = {
                'score': self.engine.score,
                'level': self.engine.level,
                'time': self.engine.game_time,
                'enemies_killed': self.engine.enemies_killed,
                'items_collected': self.engine.items_collected,
                'max_combo': self.engine.max_combo,
                'game_name': self.app.current_game.get('name', 'Game')
            }
            Clock.schedule_once(lambda dt: self.go_to_results(), 1)
        
        return True
    
    def toggle_pause(self):
        if self.engine:
            self.engine.paused = not self.engine.paused
    
    def confirm_exit(self):
        Clock.unschedule(self.update)
        self.manager.current = 'home'
    
    def go_to_results(self):
        self.manager.current = 'results'
    
    def on_leave(self):
        Clock.unschedule(self.update)


# ==================== РЕЗУЛЬТАТЫ ====================

class ResultsScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        
        self.layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
        self.add_widget(self.layout)
    
    def on_pre_enter(self):
        self.layout.clear_widgets()
        stats = self.app.last_game_stats or {}
        
        self.layout.add_widget(Label(
            text='GAME OVER',
            font_size=sp(32),
            size_hint=(1, 0.1),
            color=(1, 0.5, 0.5, 1),
            bold=True
        ))
        
        self.layout.add_widget(Label(
            text=stats.get('game_name', 'Unknown'),
            font_size=sp(16),
            size_hint=(1, 0.06),
            color=(0.7, 0.7, 0.9, 1)
        ))
        
        self.layout.add_widget(Label(
            text=f"SCORE: {stats.get('score', 0)}",
            font_size=sp(36),
            size_hint=(1, 0.12),
            color=(1, 1, 0.3, 1),
            bold=True
        ))
        
        stats_box = GridLayout(cols=2, size_hint=(1, 0.3), spacing=dp(10), padding=dp(10))
        
        stat_items = [
            ('Level', stats.get('level', 1)),
            ('Time', f"{int(stats.get('time', 0))}s"),
            ('Enemies', stats.get('enemies_killed', 0)),
            ('Items', stats.get('items_collected', 0)),
            ('Max Combo', f"x{stats.get('max_combo', 0)}"),
        ]
        
        for name, value in stat_items:
            stats_box.add_widget(Label(text=name, font_size=sp(13), color=(0.6, 0.6, 0.7, 1)))
            stats_box.add_widget(Label(text=str(value), font_size=sp(15), color=(0.9, 0.9, 1, 1), bold=True))
        
        self.layout.add_widget(stats_box)
        
        buttons = BoxLayout(size_hint=(1, 0.15), spacing=dp(10))
        
        retry_btn = Button(
            text='PLAY AGAIN',
            font_size=sp(16),
            background_color=(0.3, 0.5, 0.3, 1),
            background_normal=''
        )
        retry_btn.bind(on_release=lambda x: self.retry())
        buttons.add_widget(retry_btn)
        
        new_btn = Button(
            text='NEW GAME',
            font_size=sp(16),
            background_color=(0.3, 0.3, 0.5, 1),
            background_normal=''
        )
        new_btn.bind(on_release=lambda x: self.new_game())
        buttons.add_widget(new_btn)
        
        self.layout.add_widget(buttons)
        
        home_btn = Button(
            text='HOME',
            font_size=sp(14),
            size_hint=(1, 0.08),
            background_color=(0.3, 0.3, 0.35, 1),
            background_normal=''
        )
        home_btn.bind(on_release=lambda x: self.go_home())
        self.layout.add_widget(home_btn)
    
    def retry(self):
        self.manager.current = 'game'
    
    def new_game(self):
        self.manager.current = 'home'
    
    def go_home(self):
        self.manager.current = 'home'


# ==================== ИСТОРИЯ ====================

class HistoryScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        
        layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(10))
        
        layout.add_widget(Label(
            text='MY AI GAMES',
            font_size=sp(24),
            size_hint=(1, 0.08),
            color=(0.5, 0.8, 1, 1),
            bold=True
        ))
        
        self.scroll = ScrollView(size_hint=(1, 0.8))
        self.grid = GridLayout(cols=1, spacing=dp(8), size_hint_y=None, padding=dp(5))
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.scroll.add_widget(self.grid)
        layout.add_widget(self.scroll)
        
        back_btn = Button(
            text='< BACK',
            font_size=sp(16),
            size_hint=(1, 0.08),
            background_color=(0.3, 0.3, 0.35, 1),
            background_normal=''
        )
        back_btn.bind(on_release=lambda x: self.go_back())
        layout.add_widget(back_btn)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.refresh_list()
    
    def refresh_list(self):
        self.grid.clear_widgets()
        history = self.app.load_history()
        
        if not history:
            self.grid.add_widget(Label(
                text='No games yet!\nGenerate your first AI game.',
                font_size=sp(16),
                size_hint_y=None,
                height=dp(80)
            ))
            return
        
        for game in reversed(history[-20:]):
            desc = game.get('description', '')[:50]
            
            btn = Button(
                text=f"{game.get('name', 'Unknown')}\n{desc}...",
                font_size=sp(12),
                size_hint_y=None,
                height=dp(60),
                background_color=(0.15, 0.2, 0.25, 1),
                background_normal='',
                halign='left'
            )
            btn.bind(on_release=lambda x, g=game: self.play_game(g))
            self.grid.add_widget(btn)
    
    def play_game(self, game):
        self.app.current_game = game
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
        self.ai_generated_config = None
        self.last_game_stats = None
        self.data_path = get_data_path()
        
        # Загружаем сохранённые настройки
        saved_key = self.load_api_key()
        if saved_key:
            self.groq_client.set_api_key(saved_key)
        
        saved_model = self.load_selected_model()
        if saved_model:
            self.groq_client.set_model(saved_model)
        
        self.sm = ScreenManager(transition=FadeTransition())
        
        self.sm.add_widget(HomeScreen(self, name='home'))
        self.sm.add_widget(SettingsScreen(self, name='settings'))
        self.sm.add_widget(LoadingScreen(self, name='loading'))
        self.sm.add_widget(GameScreen(self, name='game'))
        self.sm.add_widget(ResultsScreen(self, name='results'))
        self.sm.add_widget(HistoryScreen(self, name='history'))
        
        return self.sm
    
    def build_game_config(self, ai_config):
        """Строит полный конфиг игры из ответа AI"""
        # Значения по умолчанию
        default_bg = [[0.1, 0.1, 0.2], [0.15, 0.1, 0.25], [0.05, 0.05, 0.1]]
        default_player = {'shape': 'circle', 'color': [0.3, 0.7, 1], 'speed': 5, 'health': 3}
        default_enemies = [{'name': 'Enemy', 'shape': 'circle', 'color': [1, 0.3, 0.3], 
                           'speed': 1, 'health': 1, 'points': 10, 'behavior': 'chase'}]
        default_items = [{'name': 'Health', 'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1}]
        
        return {
            'name': ai_config.get('name', 'AI Generated Game'),
            'description': ai_config.get('description', ''),
            'theme': ai_config.get('theme', 'space'),
            'game_type': ai_config.get('game_type', 'shooter'),
            'player': ai_config.get('player', default_player),
            'enemies': ai_config.get('enemies', default_enemies),
            'items': ai_config.get('items', default_items),
            'background_colors': ai_config.get('background_colors', default_bg),
            'difficulty_curve': ai_config.get('difficulty_curve', 'normal'),
            'special_mechanics': ai_config.get('special_mechanics', ''),
            'created_at': time.time()
        }
    
    def save_api_key(self, key):
        try:
            path = os.path.join(self.data_path, 'groq_key.txt')
            with open(path, 'w') as f:
                f.write(key)
        except:
            pass
    
    def load_api_key(self):
        try:
            path = os.path.join(self.data_path, 'groq_key.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        except:
            pass
        return ""
    
    def save_selected_model(self, model_id):
        try:
            path = os.path.join(self.data_path, 'selected_model.txt')
            with open(path, 'w') as f:
                f.write(model_id)
        except:
            pass
    
    def load_selected_model(self):
        try:
            path = os.path.join(self.data_path, 'selected_model.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        except:
            pass
        return ""
    
    def save_game_to_history(self, game_config):
        history = self.load_history()
        history.append(game_config)
        history = history[-50:]
        
        try:
            path = os.path.join(self.data_path, 'game_history.json')
            with open(path, 'w') as f:
                json.dump(history, f)
        except:
            pass
    
    def load_history(self):
        try:
            path = os.path.join(self.data_path, 'game_history.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []


if __name__ == '__main__':
    GameGeneratorApp().run()

"""
AI GAME GENERATOR with Groq API
Стабильная версия для Android
"""

import json
import random
import math
import time
import os
import re
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
from kivy.clock import Clock, mainthread
from kivy.animation import Animation
from kivy.metrics import dp, sp
from kivy.utils import platform
from kivy.core.window import Window
from kivy.properties import NumericProperty, StringProperty, BooleanProperty

# Безопасный импорт requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Для Android
if platform == 'android':
    try:
        from android.permissions import request_permissions, Permission
        request_permissions([Permission.INTERNET])
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
    {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "recommended": False, "speed": "Very Fast", "quality": "Good"},
]

GAME_PROMPT = """You are a game designer. Create a mini-game based on this idea: {prompt}

Return ONLY valid JSON (no markdown, no explanation):
{{
    "name": "Game Name",
    "theme": "space",
    "game_type": "shooter",
    "description": "Short description",
    "player": {{
        "shape": "triangle",
        "color": [0.3, 0.7, 1.0],
        "speed": 5,
        "health": 3
    }},
    "enemies": [
        {{"name": "Enemy1", "shape": "circle", "color": [1, 0.3, 0.3], "speed": 1.5, "health": 1, "points": 10, "behavior": "chase"}}
    ],
    "items": [
        {{"name": "Health", "color": [0.3, 1, 0.3], "effect": "heal", "value": 1}}
    ],
    "background_colors": [[0.1, 0.1, 0.2], [0.15, 0.1, 0.25]],
    "difficulty_curve": "normal"
}}"""

# Локальные шаблоны (fallback если нет API)
LOCAL_THEMES = {
    'space': {
        'bg': [[0.05, 0.05, 0.15], [0.1, 0.05, 0.2]],
        'player_color': [0.3, 0.8, 1],
        'enemy_colors': [[1, 0.3, 0.3], [1, 0.5, 0.2], [0.8, 0.2, 0.8]],
        'names': ['Cosmic', 'Stellar', 'Nebula', 'Galaxy', 'Void'],
    },
    'fantasy': {
        'bg': [[0.1, 0.15, 0.1], [0.15, 0.2, 0.1]],
        'player_color': [0.9, 0.7, 0.3],
        'enemy_colors': [[0.5, 0.2, 0.5], [0.3, 0.5, 0.2], [0.6, 0.3, 0.3]],
        'names': ['Dragon', 'Magic', 'Quest', 'Kingdom', 'Legend'],
    },
    'cyber': {
        'bg': [[0.02, 0.02, 0.08], [0.08, 0.02, 0.12]],
        'player_color': [0, 1, 1],
        'enemy_colors': [[1, 0, 0.5], [1, 0.5, 0], [0.5, 0, 1]],
        'names': ['Neon', 'Cyber', 'Hack', 'Digital', 'Matrix'],
    },
    'nature': {
        'bg': [[0.1, 0.2, 0.1], [0.15, 0.25, 0.1]],
        'player_color': [0.4, 0.8, 0.3],
        'enemy_colors': [[0.6, 0.4, 0.2], [0.4, 0.4, 0.3], [0.5, 0.3, 0.2]],
        'names': ['Wild', 'Forest', 'Jungle', 'Safari', 'Nature'],
    },
    'ocean': {
        'bg': [[0.02, 0.1, 0.2], [0.05, 0.15, 0.25]],
        'player_color': [0.2, 0.8, 0.7],
        'enemy_colors': [[0.5, 0.3, 0.5], [0.3, 0.3, 0.5], [0.6, 0.2, 0.3]],
        'names': ['Deep', 'Ocean', 'Aqua', 'Marine', 'Coral'],
    },
}

FUN_FACTS = [
    "Groq LPU processes 500+ tokens/sec!",
    "First video game: 1958",
    "Pac-Man inspired by pizza",
    "Mario was called Jumpman",
    "Tetris made in USSR",
    "Minecraft world 8x Earth size",
    "Game Boy survived a bombing",
    "AI analyzing your idea...",
    "Creating unique world...",
    "Designing enemies...",
    "Almost ready...",
]


# ==================== УТИЛИТЫ ====================

def get_data_path():
    if platform == 'android':
        try:
            from android.storage import app_storage_path
            return app_storage_path()
        except:
            return '/data/data/org.groq.aigamegen/files'
    return os.path.expanduser('~')


# ==================== GROQ API ====================

class GroqClient:
    def __init__(self):
        self.api_key = ""
        self.model = "llama-3.1-8b-instant"
        self.is_processing = False
        self.result = None
        self.error = None
    
    def set_api_key(self, key):
        self.api_key = key.strip()
    
    def set_model(self, model_id):
        self.model = model_id
    
    def generate_game_sync(self, prompt):
        """Синхронная генерация (вызывается из Clock.schedule)"""
        if not HAS_REQUESTS:
            return False, "requests library not available"
        
        if not self.api_key:
            return False, "API key not set"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a game designer. Return only valid JSON."},
                    {"role": "user", "content": GAME_PROMPT.format(prompt=prompt)}
                ],
                "temperature": 0.8,
                "max_tokens": 1500
            }
            
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 401:
                return False, "Invalid API Key"
            elif response.status_code != 200:
                return False, f"API Error: {response.status_code}"
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Извлекаем JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                game_config = json.loads(json_match.group())
                return True, game_config
            else:
                return False, "Invalid AI response format"
                
        except requests.exceptions.Timeout:
            return False, "Request timeout"
        except requests.exceptions.ConnectionError:
            return False, "No internet connection"
        except json.JSONDecodeError:
            return False, "Failed to parse AI response"
        except Exception as e:
            return False, str(e)[:50]
    
    def test_connection_sync(self):
        """Синхронный тест соединения"""
        if not HAS_REQUESTS:
            return False, "Install 'requests' library"
        
        if not self.api_key:
            return False, "Enter API key first"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 401:
                return False, "Invalid API Key"
            elif response.status_code == 200:
                return True, "Connected!"
            else:
                return False, f"Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, "No internet"
        except Exception as e:
            return False, str(e)[:30]


def generate_local_game(prompt):
    """Локальная генерация без API"""
    prompt_lower = prompt.lower()
    
    # Определяем тему
    theme = 'space'
    for theme_name in LOCAL_THEMES.keys():
        if theme_name in prompt_lower:
            theme = theme_name
            break
    
    # Ключевые слова
    if any(w in prompt_lower for w in ['ocean', 'sea', 'fish', 'water', 'море', 'океан']):
        theme = 'ocean'
    elif any(w in prompt_lower for w in ['forest', 'nature', 'animal', 'лес', 'природа']):
        theme = 'nature'
    elif any(w in prompt_lower for w in ['cyber', 'neon', 'robot', 'future', 'кибер', 'неон']):
        theme = 'cyber'
    elif any(w in prompt_lower for w in ['magic', 'dragon', 'fantasy', 'wizard', 'магия', 'дракон']):
        theme = 'fantasy'
    
    theme_data = LOCAL_THEMES[theme]
    
    # Тип игры
    game_type = 'shooter'
    if any(w in prompt_lower for w in ['collect', 'catch', 'собирать', 'ловить']):
        game_type = 'collector'
    elif any(w in prompt_lower for w in ['avoid', 'dodge', 'escape', 'избегать', 'убегать']):
        game_type = 'avoider'
    elif any(w in prompt_lower for w in ['survive', 'survival', 'выживание', 'выжить']):
        game_type = 'survival'
    
    # Генерируем название
    name_prefix = random.choice(['Super', 'Ultra', 'Mega', 'Epic', 'Turbo', 'Hyper'])
    name_middle = random.choice(theme_data['names'])
    name_suffix = random.choice(['Quest', 'Rush', 'Blitz', 'Force', 'Strike', 'Arena'])
    game_name = f"{name_prefix} {name_middle} {name_suffix}"
    
    # Генерируем врагов
    enemies = []
    behaviors = ['chase', 'patrol', 'zigzag', 'spiral']
    shapes = ['circle', 'square', 'triangle']
    
    for i, color in enumerate(theme_data['enemy_colors'][:3]):
        enemies.append({
            'name': f"Enemy {i+1}",
            'shape': random.choice(shapes),
            'color': color,
            'speed': 0.8 + i * 0.4,
            'health': 1 + i,
            'points': 10 + i * 15,
            'behavior': behaviors[i % len(behaviors)]
        })
    
    # Генерируем предметы
    items = [
        {'name': 'Health Pack', 'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1},
        {'name': 'Shield', 'color': [0.3, 0.7, 1], 'effect': 'shield', 'value': 3},
        {'name': 'Speed Boost', 'color': [1, 1, 0.3], 'effect': 'speed', 'value': 5},
        {'name': 'Magnet', 'color': [1, 0.5, 1], 'effect': 'magnet', 'value': 5},
        {'name': 'Double Points', 'color': [1, 0.8, 0.2], 'effect': 'double_points', 'value': 10},
    ]
    
    return {
        'name': game_name,
        'theme': theme,
        'game_type': game_type,
        'description': f"AI-generated {theme} {game_type} game",
        'player': {
            'shape': 'triangle',
            'color': theme_data['player_color'],
            'speed': 5,
            'health': 3
        },
        'enemies': enemies,
        'items': items,
        'background_colors': theme_data['bg'],
        'difficulty_curve': 'normal',
        'created_at': time.time(),
        'ai_generated': False
    }


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
        super().__init__(x, y, dp(30), color, config.get('shape', 'triangle'))
        self.max_health = config.get('health', 3)
        self.health = self.max_health
        self.speed = config.get('speed', 5)
        self.shield = 0
        self.magnet = 0
        self.double_points = 0
        self.invincible = 0
        self.trail = []
    
    def update(self, dt):
        super().update(dt)
        self.shield = max(0, self.shield - dt)
        self.magnet = max(0, self.magnet - dt)
        self.double_points = max(0, self.double_points - dt)
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
        self.behavior = config.get('behavior', 'chase')
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
        else:
            self.vy = -self.speed * dp(40)
        
        super().update(dt)
    
    def take_damage(self, amount=1):
        self.health -= amount
        return self.health <= 0


class Item(GameObject):
    def __init__(self, x, y, config):
        color = config.get('color', [1, 1, 0.3])
        super().__init__(x, y, dp(20), color, 'circle')
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
        self.stars = []
        for _ in range(50):
            self.stars.append({
                'x': random.uniform(0, max(100, self.width)),
                'y': random.uniform(0, max(100, self.height)),
                'size': random.uniform(1, 3),
                'speed': random.uniform(20, 80),
                'brightness': random.uniform(0.3, 1.0)
            })
    
    def start_game(self):
        self.setup_stars()
        
        player_config = self.config.get('player', {})
        self.player = Player(self.width / 2, self.height * 0.2, player_config)
        
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
        
        # Уровни
        if self.enemies_killed >= 5 + self.level * 3 and not self.enemies:
            self.level += 1
            self.enemies_killed = 0
            self.score += self.level * 100
            for _ in range(20):
                self.particles.append(Particle(
                    random.uniform(0, self.width),
                    random.uniform(0, self.height),
                    (1, 1, 0.3)
                ))
        
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
        self.player.x = max(self.player.size/2, min(self.width - self.player.size/2, self.player.x))
        self.player.y = max(self.player.size/2, min(self.height - self.player.size/2, self.player.y))
        
        self.player.update(dt)
        
        # Автострельба для shooter
        if self.config.get('game_type') == 'shooter':
            if int(self.game_time * 5) % 2 == 0 and len(self.bullets) < 10:
                self.bullets.append(Bullet(
                    self.player.x, self.player.y + self.player.size/2,
                    0, dp(400), self.player.color
                ))
        
        # Спавн врагов
        spawn_rate = 2.0 / (1 + self.level * 0.1)
        if self.spawn_timer > spawn_rate and len(self.enemies) < 12:
            self.spawn_enemy()
            self.spawn_timer = 0
        
        # Спавн предметов
        if random.random() < 0.005:
            self.spawn_item()
        
        # Обновление врагов
        for enemy in self.enemies[:]:
            enemy.update(dt, self.player.x, self.player.y)
            
            if enemy.y < -enemy.size or enemy.y > self.height + enemy.size:
                self.enemies.remove(enemy)
                continue
            
            if enemy.collides_with(self.player):
                if self.player.take_damage():
                    self.game_over = True
                else:
                    self.spawn_particles(enemy.x, enemy.y, enemy.color)
                    self.enemies.remove(enemy)
        
        # Обновление предметов
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
        
        # Обновление пуль
        for bullet in self.bullets[:]:
            bullet.update(dt)
            
            if bullet.y > self.height + bullet.size:
                self.bullets.remove(bullet)
                continue
            
            for enemy in self.enemies[:]:
                if bullet.collides_with(enemy):
                    if enemy.take_damage():
                        self.kill_enemy(enemy)
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    self.spawn_particles(bullet.x, bullet.y, bullet.color)
                    break
        
        # Обновление частиц
        self.particles = [p for p in self.particles if p.update(dt)]
        
        # Обновление звёзд
        for star in self.stars:
            star['y'] -= star['speed'] * dt
            if star['y'] < 0:
                star['y'] = self.height
                star['x'] = random.uniform(0, self.width)
        
        self.draw()
    
    def spawn_enemy(self):
        enemies_config = self.config.get('enemies', [])
        if not enemies_config:
            enemies_config = [{'color': [1, 0.3, 0.3], 'speed': 1, 'health': 1, 'points': 10}]
        
        config = random.choice(enemies_config).copy()
        config['speed'] = config.get('speed', 1) * (1 + self.level * 0.1)
        
        x = random.uniform(dp(30), self.width - dp(30))
        y = self.height + dp(30)
        
        self.enemies.append(Enemy(x, y, config))
    
    def spawn_item(self):
        items_config = self.config.get('items', [])
        if not items_config:
            items_config = [{'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1}]
        
        config = random.choice(items_config)
        x = random.uniform(dp(30), self.width - dp(30))
        y = self.height + dp(20)
        
        self.items.append(Item(x, y, config))
    
    def kill_enemy(self, enemy):
        points = enemy.points
        if self.player.double_points > 0:
            points *= 2
        
        self.combo += 1
        self.combo_timer = 2.0
        self.max_combo = max(self.max_combo, self.combo)
        points = int(points * (1 + self.combo * 0.1))
        self.score += points
        self.enemies_killed += 1
        
        self.spawn_particles(enemy.x, enemy.y, enemy.color)
        if enemy in self.enemies:
            self.enemies.remove(enemy)
    
    def collect_item(self, item):
        self.items_collected += 1
        self.score += 25
        
        if item.effect == 'heal':
            self.player.health = min(self.player.max_health, self.player.health + item.value)
        elif item.effect == 'shield':
            self.player.shield += item.value
        elif item.effect == 'speed':
            self.player.speed *= 1.2
        elif item.effect == 'magnet':
            self.player.magnet += item.value
        elif item.effect == 'double_points':
            self.player.double_points += item.value
        elif item.effect == 'clear_enemies':
            for e in self.enemies[:]:
                self.kill_enemy(e)
        
        self.spawn_particles(item.x, item.y, item.color)
    
    def spawn_particles(self, x, y, color):
        for _ in range(10):
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
            # Фон
            Color(bg[0], bg[1], bg[2], 1)
            Rectangle(pos=self.pos, size=self.size)
            
            # Звёзды
            for star in self.stars:
                b = star['brightness']
                Color(b, b, b * 1.1, b)
                Ellipse(pos=(star['x'], star['y']), size=(star['size'], star['size']))
            
            # Частицы
            for p in self.particles:
                Color(*p.color, p.life)
                Ellipse(pos=(p.x - p.size/2, p.y - p.size/2), size=(p.size, p.size))
            
            # Предметы
            for item in self.items:
                pulse = 1 + 0.2 * math.sin(item.pulse)
                size = item.size * pulse
                Color(*item.color, 0.3)
                Ellipse(pos=(item.x - size, item.y - size), size=(size * 2, size * 2))
                Color(*item.color, 1)
                Ellipse(pos=(item.x - item.size/2, item.y - item.size/2), size=(item.size, item.size))
            
            # Пули
            for bullet in self.bullets:
                Color(*bullet.color, 1)
                Ellipse(pos=(bullet.x - bullet.size/2, bullet.y - bullet.size/2), 
                       size=(bullet.size, bullet.size))
            
            # Враги
            for enemy in self.enemies:
                Color(0, 0, 0, 0.3)
                self.draw_shape(enemy.shape, enemy.x + 3, enemy.y - 3, enemy.size)
                Color(*enemy.color, 1)
                self.draw_shape(enemy.shape, enemy.x, enemy.y, enemy.size)
                
                # HP bar
                if enemy.health < enemy.max_health:
                    bw = enemy.size
                    Color(0.3, 0.3, 0.3, 0.8)
                    Rectangle(pos=(enemy.x - bw/2, enemy.y + enemy.size/2 + 5), size=(bw, dp(4)))
                    Color(0.2, 0.8, 0.2, 1)
                    Rectangle(pos=(enemy.x - bw/2, enemy.y + enemy.size/2 + 5), 
                             size=(bw * enemy.health / enemy.max_health, dp(4)))
            
            # Игрок
            if self.player:
                # Trail
                for i, (tx, ty, _) in enumerate(self.player.trail):
                    alpha = i / max(1, len(self.player.trail)) * 0.3
                    Color(*self.player.color, alpha)
                    size = self.player.size * (0.3 + 0.7 * i / max(1, len(self.player.trail)))
                    self.draw_shape(self.player.shape, tx, ty, size)
                
                # Glow
                Color(*self.player.color, 0.2)
                glow = self.player.size * 1.5
                Ellipse(pos=(self.player.x - glow/2, self.player.y - glow/2), size=(glow, glow))
                
                # Player
                if self.player.invincible > 0 and int(self.player.invincible * 10) % 2:
                    Color(*self.player.color, 0.5)
                else:
                    Color(*self.player.color, 1)
                self.draw_shape(self.player.shape, self.player.x, self.player.y, self.player.size)
                
                # Shield
                if self.player.shield > 0:
                    Color(0.3, 0.7, 1, 0.4)
                    Line(circle=(self.player.x, self.player.y, self.player.size * 0.8), width=2)
    
    def draw_shape(self, shape, x, y, size):
        half = size / 2
        if shape == 'circle':
            Ellipse(pos=(x - half, y - half), size=(size, size))
        elif shape == 'square':
            Rectangle(pos=(x - half, y - half), size=(size, size))
        elif shape == 'triangle':
            Triangle(points=[x, y + half, x - half, y - half, x + half, y - half])
        else:
            Ellipse(pos=(x - half, y - half), size=(size, size))


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
        
        layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(8))
        
        # Header
        header = BoxLayout(size_hint=(1, 0.07))
        back_btn = Button(text='<', size_hint=(0.15, 1), font_size=sp(22),
                         background_color=(0.3, 0.3, 0.4, 1), background_normal='')
        back_btn.bind(on_release=lambda x: self.go_back())
        header.add_widget(back_btn)
        header.add_widget(Label(text='API Settings', font_size=sp(20), color=(0.8, 0.8, 1, 1)))
        layout.add_widget(header)
        
        # API Key section
        layout.add_widget(Label(text='Groq API Key:', font_size=sp(14), 
                                size_hint=(1, 0.04), color=(0.5, 0.8, 1, 1)))
        
        self.api_key_input = TextInput(
            hint_text='gsk_xxxxx...',
            multiline=False,
            password=True,
            font_size=sp(14),
            size_hint=(1, 0.08),
            background_color=(0.12, 0.12, 0.18, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(0.5, 0.8, 1, 1)
        )
        saved_key = self.app.load_api_key()
        if saved_key:
            self.api_key_input.text = saved_key
        layout.add_widget(self.api_key_input)
        
        # Status
        self.status_label = Label(text='', font_size=sp(12), size_hint=(1, 0.04),
                                  color=(0.5, 0.5, 0.5, 1))
        layout.add_widget(self.status_label)
        
        # Buttons
        btn_row = BoxLayout(size_hint=(1, 0.07), spacing=dp(10))
        
        show_btn = Button(text='Show/Hide', font_size=sp(12),
                         background_color=(0.3, 0.3, 0.4, 1), background_normal='')
        show_btn.bind(on_release=lambda x: self.toggle_password())
        btn_row.add_widget(show_btn)
        
        test_btn = Button(text='Test', font_size=sp(12),
                         background_color=(0.3, 0.5, 0.6, 1), background_normal='')
        test_btn.bind(on_release=lambda x: self.test_connection())
        btn_row.add_widget(test_btn)
        
        save_btn = Button(text='Save', font_size=sp(12),
                         background_color=(0.3, 0.5, 0.3, 1), background_normal='')
        save_btn.bind(on_release=lambda x: self.save_key())
        btn_row.add_widget(save_btn)
        
        layout.add_widget(btn_row)
        
        # Get key info
        layout.add_widget(Label(text='Get FREE key: console.groq.com', font_size=sp(11),
                                size_hint=(1, 0.03), color=(0.4, 0.6, 0.8, 1)))
        
        # Model selection
        layout.add_widget(Label(text='Select Model:', font_size=sp(14),
                                size_hint=(1, 0.04), color=(0.5, 0.8, 1, 1)))
        
        # Search
        self.model_search = TextInput(
            hint_text='Search models...',
            multiline=False,
            font_size=sp(13),
            size_hint=(1, 0.06),
            background_color=(0.12, 0.12, 0.18, 1),
            foreground_color=(1, 1, 1, 1)
        )
        self.model_search.bind(text=self.filter_models)
        layout.add_widget(self.model_search)
        
        # Models list
        scroll = ScrollView(size_hint=(1, 0.4))
        self.models_grid = GridLayout(cols=1, spacing=dp(5), size_hint_y=None, padding=dp(5))
        self.models_grid.bind(minimum_height=self.models_grid.setter('height'))
        self.refresh_models()
        scroll.add_widget(self.models_grid)
        layout.add_widget(scroll)
        
        # Selected model
        self.selected_label = Label(text=f'Selected: {GROQ_MODELS[0]["name"]}',
                                    font_size=sp(13), size_hint=(1, 0.04), color=(0.5, 1, 0.5, 1))
        layout.add_widget(self.selected_label)
        
        self.add_widget(layout)
    
    def refresh_models(self, filter_text=''):
        self.models_grid.clear_widgets()
        
        for model in GROQ_MODELS:
            if filter_text and filter_text.lower() not in model['name'].lower():
                continue
            
            is_selected = model['id'] == self.selected_model
            is_rec = model['recommended']
            
            if is_rec:
                bg = (0.25, 0.35, 0.25, 1) if is_selected else (0.2, 0.28, 0.2, 1)
                prefix = "* "
            else:
                bg = (0.25, 0.25, 0.35, 1) if is_selected else (0.15, 0.15, 0.2, 1)
                prefix = "  "
            
            btn = Button(
                text=f"{prefix}{model['name']}\n   {model['speed']} | {model['quality']}",
                font_size=sp(11),
                size_hint_y=None,
                height=dp(50),
                background_color=bg,
                background_normal='',
                halign='left'
            )
            btn.bind(on_release=lambda x, m=model: self.select_model(m))
            self.models_grid.add_widget(btn)
    
    def filter_models(self, instance, text):
        self.refresh_models(text)
    
    def select_model(self, model):
        self.selected_model = model['id']
        self.selected_label.text = f"Selected: {model['name']}"
        self.app.groq_client.set_model(model['id'])
        self.app.save_selected_model(model['id'])
        self.refresh_models(self.model_search.text)
    
    def toggle_password(self):
        self.api_key_input.password = not self.api_key_input.password
    
    def save_key(self):
        key = self.api_key_input.text.strip()
        self.app.save_api_key(key)
        self.app.groq_client.set_api_key(key)
        self.status_label.text = 'Key saved!'
        self.status_label.color = (0.5, 1, 0.5, 1)
    
    def test_connection(self):
        key = self.api_key_input.text.strip()
        if not key:
            self.status_label.text = 'Enter API key first'
            self.status_label.color = (1, 0.5, 0.5, 1)
            return
        
        self.app.groq_client.set_api_key(key)
        self.status_label.text = 'Testing...'
        self.status_label.color = (1, 1, 0.5, 1)
        
        # Используем Clock для выполнения запроса
        Clock.schedule_once(lambda dt: self._do_test(), 0.1)
    
    def _do_test(self):
        success, message = self.app.groq_client.test_connection_sync()
        self.status_label.text = message
        self.status_label.color = (0.5, 1, 0.5, 1) if success else (1, 0.5, 0.5, 1)
    
    def go_back(self):
        self.manager.transition = SlideTransition(direction='right')
        self.manager.current = 'home'


class HomeScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        
        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(8))
        
        # Title
        layout.add_widget(Label(text='AI GAME', font_size=sp(40), size_hint=(1, 0.1),
                                bold=True, color=(0.5, 0.8, 1, 1)))
        layout.add_widget(Label(text='GENERATOR', font_size=sp(32), size_hint=(1, 0.08),
                                bold=True, color=(1, 0.5, 0.8, 1)))
        layout.add_widget(Label(text='Powered by Groq AI', font_size=sp(11),
                                size_hint=(1, 0.03), color=(0.5, 0.5, 0.6, 1)))
        
        # API Status
        self.api_status = Label(text='API: Not configured', font_size=sp(11),
                                size_hint=(1, 0.04), color=(1, 0.5, 0.5, 1))
        layout.add_widget(self.api_status)
        
        # Prompt input
        layout.add_widget(Label(text='Describe your game:', font_size=sp(13),
                                size_hint=(1, 0.04), color=(0.7, 0.7, 0.8, 1)))
        
        self.prompt_input = TextInput(
            hint_text='Example: Space shooter with neon effects and alien enemies',
            multiline=True,
            font_size=sp(13),
            size_hint=(1, 0.16),
            background_color=(0.1, 0.1, 0.15, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(0.5, 0.8, 1, 1)
        )
        layout.add_widget(self.prompt_input)
        
        # Quick examples
        examples = GridLayout(cols=2, size_hint=(1, 0.1), spacing=dp(5))
        for name, prompt in [('Space', 'space shooter aliens'), ('Fantasy', 'magic dragon quest'),
                             ('Ocean', 'underwater fish adventure'), ('Cyber', 'neon robot battle')]:
            btn = Button(text=name, font_size=sp(11), background_color=(0.2, 0.2, 0.3, 1), background_normal='')
            btn.bind(on_release=lambda x, p=prompt: setattr(self.prompt_input, 'text', p))
            examples.add_widget(btn)
        layout.add_widget(examples)
        
        # Generate button
        self.gen_btn = Button(text='GENERATE GAME', font_size=sp(18), size_hint=(1, 0.1),
                              background_color=(0.3, 0.6, 0.3, 1), background_normal='', bold=True)
        self.gen_btn.bind(on_release=lambda x: self.generate())
        layout.add_widget(self.gen_btn)
        
        # Local generate button
        self.local_btn = Button(text='Quick Generate (Offline)', font_size=sp(13), size_hint=(1, 0.06),
                                background_color=(0.4, 0.3, 0.5, 1), background_normal='')
        self.local_btn.bind(on_release=lambda x: self.generate_local())
        layout.add_widget(self.local_btn)
        
        # Bottom buttons
        bottom = BoxLayout(size_hint=(1, 0.07), spacing=dp(10))
        
        settings_btn = Button(text='API Settings', font_size=sp(13),
                              background_color=(0.35, 0.3, 0.45, 1), background_normal='')
        settings_btn.bind(on_release=lambda x: self.go_settings())
        bottom.add_widget(settings_btn)
        
        history_btn = Button(text='My Games', font_size=sp(13),
                             background_color=(0.25, 0.25, 0.35, 1), background_normal='')
        history_btn.bind(on_release=lambda x: self.go_history())
        bottom.add_widget(history_btn)
        
        layout.add_widget(bottom)
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.update_status()
    
    def update_status(self):
        key = self.app.load_api_key()
        if key and HAS_REQUESTS:
            self.api_status.text = f'API: Ready | Model: {self.app.groq_client.model}'
            self.api_status.color = (0.5, 1, 0.5, 1)
            self.gen_btn.background_color = (0.3, 0.6, 0.3, 1)
        elif not HAS_REQUESTS:
            self.api_status.text = 'API: Offline mode only'
            self.api_status.color = (1, 0.7, 0.3, 1)
            self.gen_btn.background_color = (0.4, 0.4, 0.4, 1)
        else:
            self.api_status.text = 'API: Not configured - Go to Settings'
            self.api_status.color = (1, 0.5, 0.5, 1)
            self.gen_btn.background_color = (0.4, 0.4, 0.4, 1)
    
    def generate(self):
        key = self.app.load_api_key()
        if not key or not HAS_REQUESTS:
            self.api_status.text = 'No API key! Using offline mode...'
            self.api_status.color = (1, 0.7, 0.3, 1)
            Clock.schedule_once(lambda dt: self.generate_local(), 0.5)
            return
        
        prompt = self.prompt_input.text.strip() or "fun arcade game"
        self.app.current_prompt = prompt
        self.manager.current = 'loading'
    
    def generate_local(self):
        prompt = self.prompt_input.text.strip() or "space shooter game"
        self.app.current_game = generate_local_game(prompt)
        self.app.save_game_to_history(self.app.current_game)
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
        self.dots = 0
        self.api_done = False
        self.api_result = None
        self.api_error = None
        
        layout = FloatLayout()
        
        self.title_label = Label(text='AI', font_size=sp(48), bold=True,
                                 pos_hint={'center_x': 0.5, 'center_y': 0.75},
                                 color=(0.5, 0.8, 1, 1))
        layout.add_widget(self.title_label)
        
        self.model_label = Label(text='', font_size=sp(11),
                                 pos_hint={'center_x': 0.5, 'center_y': 0.68},
                                 color=(0.5, 0.5, 0.6, 1))
        layout.add_widget(self.model_label)
        
        self.progress_bar = ProgressBar(max=100, value=0, size_hint=(0.8, 0.03),
                                        pos_hint={'center_x': 0.5, 'center_y': 0.55})
        layout.add_widget(self.progress_bar)
        
        self.percent_label = Label(text='0%', font_size=sp(16),
                                   pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                   color=(0.5, 0.8, 1, 1))
        layout.add_widget(self.percent_label)
        
        self.status_label = Label(text='Initializing...', font_size=sp(13),
                                  pos_hint={'center_x': 0.5, 'center_y': 0.43},
                                  color=(0.6, 0.6, 0.7, 1))
        layout.add_widget(self.status_label)
        
        self.fact_label = Label(text='', font_size=sp(12),
                                pos_hint={'center_x': 0.5, 'center_y': 0.25},
                                color=(0.6, 0.6, 0.5, 1), halign='center',
                                size_hint=(0.9, 0.15))
        self.fact_label.bind(size=lambda *x: setattr(self.fact_label, 'text_size', 
                                                      (self.fact_label.width, None)))
        layout.add_widget(self.fact_label)
        
        self.error_label = Label(text='', font_size=sp(11),
                                 pos_hint={'center_x': 0.5, 'center_y': 0.15},
                                 color=(1, 0.5, 0.5, 1), halign='center',
                                 size_hint=(0.9, 0.1))
        layout.add_widget(self.error_label)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.progress = 0
        self.api_done = False
        self.api_result = None
        self.api_error = None
        self.error_label.text = ''
        self.model_label.text = f'Using: {self.app.groq_client.model}'
        self.fact_label.text = random.choice(FUN_FACTS)
        
        Clock.schedule_interval(self.update_progress, 0.05)
        Clock.schedule_once(self.start_api_call, 0.2)
    
    def start_api_call(self, dt):
        # Делаем API запрос
        Clock.schedule_once(self._do_api_call, 0.1)
    
    def _do_api_call(self, dt):
        success, result = self.app.groq_client.generate_game_sync(self.app.current_prompt)
        if success:
            self.api_result = result
        else:
            self.api_error = result
        self.api_done = True
    
    def update_progress(self, dt):
        self.dots += dt
        
        # Проверяем ошибку
        if self.api_error:
            Clock.unschedule(self.update_progress)
            self.error_label.text = f"Error: {self.api_error}\nUsing offline mode..."
            self.status_label.text = "Switching to local generation..."
            Clock.schedule_once(self.use_local_fallback, 2)
            return False
        
        # Проверяем успех
        if self.api_done and self.api_result:
            self.progress = 100
            self.progress_bar.value = 100
            self.percent_label.text = "100%"
            self.status_label.text = "Game Ready!"
            Clock.unschedule(self.update_progress)
            Clock.schedule_once(self.finish, 0.5)
            return False
        
        # Анимация прогресса (до 90%)
        if self.progress < 90:
            self.progress += random.uniform(0.5, 1.5)
        
        self.progress_bar.value = self.progress
        self.percent_label.text = f"{int(self.progress)}%"
        
        dots = '.' * (int(self.dots * 2) % 4)
        statuses = ['Connecting', 'Analyzing', 'Creating', 'Designing', 'Building']
        idx = min(int(self.progress / 20), len(statuses) - 1)
        self.status_label.text = statuses[idx] + dots
        
        # AI pulse
        pulse = 1 + 0.1 * math.sin(self.dots * 5)
        self.title_label.font_size = sp(48 * pulse)
        
        # Update fact
        if int(self.dots) % 4 == 0:
            self.fact_label.text = random.choice(FUN_FACTS)
        
        return True
    
    def use_local_fallback(self, dt):
        self.app.current_game = generate_local_game(self.app.current_prompt)
        self.app.save_game_to_history(self.app.current_game)
        self.manager.current = 'game'
    
    def finish(self, dt):
        self.app.current_game = self.app.build_game_config(self.api_result)
        self.app.save_game_to_history(self.app.current_game)
        self.manager.current = 'game'
    
    def on_leave(self):
        Clock.unschedule(self.update_progress)


class GameScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.engine = None
        self.update_event = None
        self.layout = FloatLayout()
        self.add_widget(self.layout)
    
    def on_pre_enter(self):
        self.setup_game()
    
    def setup_game(self):
        self.layout.clear_widgets()
        
        if not self.app.current_game:
            return
        
        # Engine
        self.engine = GameEngine(self.app.current_game, size_hint=(1, 0.85))
        self.layout.add_widget(self.engine)
        
        # Top panel
        top = BoxLayout(size_hint=(1, 0.08), pos_hint={'top': 1}, padding=dp(5), spacing=dp(5))
        with top.canvas.before:
            Color(0, 0, 0, 0.5)
            self.top_bg = Rectangle(pos=top.pos, size=top.size)
        top.bind(pos=lambda *x: setattr(self.top_bg, 'pos', top.pos),
                size=lambda *x: setattr(self.top_bg, 'size', top.size))
        
        self.name_lbl = Label(text=self.app.current_game.get('name', 'Game')[:20],
                              font_size=sp(12), size_hint=(0.4, 1), color=(0.8, 0.8, 1, 1))
        top.add_widget(self.name_lbl)
        
        self.score_lbl = Label(text='Score: 0', font_size=sp(14), 
                               size_hint=(0.35, 1), color=(1, 1, 0.3, 1))
        top.add_widget(self.score_lbl)
        
        self.level_lbl = Label(text='Lv.1', font_size=sp(12),
                               size_hint=(0.25, 1), color=(0.5, 1, 0.5, 1))
        top.add_widget(self.level_lbl)
        
        self.layout.add_widget(top)
        
        # Health bar
        health_panel = BoxLayout(size_hint=(1, 0.04), pos_hint={'top': 0.92}, padding=(dp(10), 0))
        self.health_lbl = Label(text='HP:', font_size=sp(11), size_hint=(0.12, 1), color=(1, 0.5, 0.5, 1))
        health_panel.add_widget(self.health_lbl)
        
        hp = self.app.current_game.get('player', {}).get('health', 3)
        self.health_bar = ProgressBar(max=hp, value=hp, size_hint=(0.88, 0.6))
        health_panel.add_widget(self.health_bar)
        self.layout.add_widget(health_panel)
        
        # Bottom panel
        bottom = BoxLayout(size_hint=(1, 0.07), pos_hint={'y': 0}, spacing=dp(5), padding=dp(5))
        with bottom.canvas.before:
            Color(0, 0, 0, 0.5)
            self.bot_bg = Rectangle(pos=bottom.pos, size=bottom.size)
        bottom.bind(pos=lambda *x: setattr(self.bot_bg, 'pos', bottom.pos),
                   size=lambda *x: setattr(self.bot_bg, 'size', bottom.size))
        
        pause_btn = Button(text='||', font_size=sp(16), size_hint=(0.15, 1),
                          background_color=(0.4, 0.4, 0.5, 1), background_normal='')
        pause_btn.bind(on_release=lambda x: self.toggle_pause())
        bottom.add_widget(pause_btn)
        
        self.combo_lbl = Label(text='', font_size=sp(13), size_hint=(0.4, 1), color=(1, 0.8, 0.2, 1))
        bottom.add_widget(self.combo_lbl)
        
        self.time_lbl = Label(text='0:00', font_size=sp(13), size_hint=(0.25, 1), color=(0.7, 0.7, 0.8, 1))
        bottom.add_widget(self.time_lbl)
        
        exit_btn = Button(text='X', font_size=sp(14), size_hint=(0.15, 1),
                         background_color=(0.5, 0.3, 0.3, 1), background_normal='')
        exit_btn.bind(on_release=lambda x: self.exit_game())
        bottom.add_widget(exit_btn)
        
        self.layout.add_widget(bottom)
        
        # Start
        self.engine.start_game()
        self.update_event = Clock.schedule_interval(self.update, 1/60)
    
    def update(self, dt):
        if not self.engine:
            return False
        
        self.engine.update(dt)
        
        self.score_lbl.text = f"Score: {self.engine.score}"
        self.level_lbl.text = f"Lv.{self.engine.level}"
        
        if self.engine.player:
            self.health_bar.value = self.engine.player.health
            self.combo_lbl.text = f"x{self.engine.combo}" if self.engine.combo > 1 else ""
        
        m, s = divmod(int(self.engine.game_time), 60)
        self.time_lbl.text = f"{m}:{s:02d}"
        
        if self.engine.game_over:
            if self.update_event:
                self.update_event.cancel()
            self.app.last_stats = {
                'score': self.engine.score,
                'level': self.engine.level,
                'time': self.engine.game_time,
                'enemies': self.engine.enemies_killed,
                'items': self.engine.items_collected,
                'combo': self.engine.max_combo,
                'name': self.app.current_game.get('name', 'Game')
            }
            Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'results'), 1)
            return False
        
        return True
    
    def toggle_pause(self):
        if self.engine:
            self.engine.paused = not self.engine.paused
    
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
        self.layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
        self.add_widget(self.layout)
    
    def on_pre_enter(self):
        self.layout.clear_widgets()
        stats = self.app.last_stats or {}
        
        self.layout.add_widget(Label(text='GAME OVER', font_size=sp(30), size_hint=(1, 0.1),
                                     color=(1, 0.5, 0.5, 1), bold=True))
        
        self.layout.add_widget(Label(text=stats.get('name', '')[:25], font_size=sp(14),
                                     size_hint=(1, 0.05), color=(0.7, 0.7, 0.9, 1)))
        
        self.layout.add_widget(Label(text=f"SCORE: {stats.get('score', 0)}", font_size=sp(32),
                                     size_hint=(1, 0.12), color=(1, 1, 0.3, 1), bold=True))
        
        # Stats grid
        grid = GridLayout(cols=2, size_hint=(1, 0.3), spacing=dp(8), padding=dp(10))
        for name, key in [('Level', 'level'), ('Time', 'time'), ('Enemies', 'enemies'), 
                          ('Items', 'items'), ('Max Combo', 'combo')]:
            val = stats.get(key, 0)
            if key == 'time':
                val = f"{int(val)}s"
            elif key == 'combo':
                val = f"x{val}"
            grid.add_widget(Label(text=name, font_size=sp(12), color=(0.6, 0.6, 0.7, 1)))
            grid.add_widget(Label(text=str(val), font_size=sp(14), color=(0.9, 0.9, 1, 1), bold=True))
        self.layout.add_widget(grid)
        
        # Buttons
        btns = BoxLayout(size_hint=(1, 0.12), spacing=dp(10))
        
        retry = Button(text='RETRY', font_size=sp(15), background_color=(0.3, 0.5, 0.3, 1), background_normal='')
        retry.bind(on_release=lambda x: setattr(self.manager, 'current', 'game'))
        btns.add_widget(retry)
        
        new_game = Button(text='NEW', font_size=sp(15), background_color=(0.3, 0.3, 0.5, 1), background_normal='')
        new_game.bind(on_release=lambda x: setattr(self.manager, 'current', 'home'))
        btns.add_widget(new_game)
        
        self.layout.add_widget(btns)
        
        home = Button(text='HOME', font_size=sp(13), size_hint=(1, 0.08),
                     background_color=(0.3, 0.3, 0.35, 1), background_normal='')
        home.bind(on_release=lambda x: setattr(self.manager, 'current', 'home'))
        self.layout.add_widget(home)


class HistoryScreen(BaseScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        
        layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(8))
        
        layout.add_widget(Label(text='MY GAMES', font_size=sp(22), size_hint=(1, 0.08),
                                color=(0.5, 0.8, 1, 1), bold=True))
        
        scroll = ScrollView(size_hint=(1, 0.82))
        self.grid = GridLayout(cols=1, spacing=dp(6), size_hint_y=None, padding=dp(5))
        self.grid.bind(minimum_height=self.grid.setter('height'))
        scroll.add_widget(self.grid)
        layout.add_widget(scroll)
        
        back = Button(text='< BACK', font_size=sp(14), size_hint=(1, 0.07),
                     background_color=(0.3, 0.3, 0.35, 1), background_normal='')
        back.bind(on_release=lambda x: self.go_back())
        layout.add_widget(back)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.refresh()
    
    def refresh(self):
        self.grid.clear_widgets()
        history = self.app.load_history()
        
        if not history:
            self.grid.add_widget(Label(text='No games yet!\nGenerate your first game.',
                                       font_size=sp(14), size_hint_y=None, height=dp(60)))
            return
        
        for game in reversed(history[-15:]):
            ai_tag = "[AI]" if game.get('ai_generated', True) else "[Local]"
            name = game.get('name', 'Unknown')[:20]
            
            btn = Button(
                text=f"{ai_tag} {name}\n{game.get('description', '')[:40]}...",
                font_size=sp(11),
                size_hint_y=None,
                height=dp(55),
                background_color=(0.15, 0.18, 0.22, 1),
                background_normal='',
                halign='left'
            )
            btn.bind(on_release=lambda x, g=game: self.play(g))
            self.grid.add_widget(btn)
    
    def play(self, game):
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
        self.last_stats = None
        self.data_path = get_data_path()
        
        # Load saved settings
        key = self.load_api_key()
        if key:
            self.groq_client.set_api_key(key)
        
        model = self.load_selected_model()
        if model:
            self.groq_client.set_model(model)
        
        # Screens
        self.sm = ScreenManager(transition=FadeTransition())
        self.sm.add_widget(HomeScreen(self, name='home'))
        self.sm.add_widget(SettingsScreen(self, name='settings'))
        self.sm.add_widget(LoadingScreen(self, name='loading'))
        self.sm.add_widget(GameScreen(self, name='game'))
        self.sm.add_widget(ResultsScreen(self, name='results'))
        self.sm.add_widget(HistoryScreen(self, name='history'))
        
        return self.sm
    
    def build_game_config(self, ai_config):
        default_bg = [[0.1, 0.1, 0.2], [0.15, 0.1, 0.25]]
        default_player = {'shape': 'triangle', 'color': [0.3, 0.7, 1], 'speed': 5, 'health': 3}
        default_enemies = [{'name': 'Enemy', 'color': [1, 0.3, 0.3], 'speed': 1, 'health': 1, 'points': 10}]
        default_items = [{'name': 'Health', 'color': [0.3, 1, 0.3], 'effect': 'heal', 'value': 1}]
        
        return {
            'name': ai_config.get('name', 'AI Game'),
            'description': ai_config.get('description', 'AI generated game'),
            'theme': ai_config.get('theme', 'space'),
            'game_type': ai_config.get('game_type', 'shooter'),
            'player': ai_config.get('player', default_player),
            'enemies': ai_config.get('enemies', default_enemies),
            'items': ai_config.get('items', default_items),
            'background_colors': ai_config.get('background_colors', default_bg),
            'difficulty_curve': ai_config.get('difficulty_curve', 'normal'),
            'created_at': time.time(),
            'ai_generated': True
        }
    
    def save_api_key(self, key):
        try:
            path = os.path.join(self.data_path, 'api_key.txt')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(key)
        except:
            pass
    
    def load_api_key(self):
        try:
            path = os.path.join(self.data_path, 'api_key.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        except:
            pass
        return ""
    
    def save_selected_model(self, model_id):
        try:
            path = os.path.join(self.data_path, 'model.txt')
            with open(path, 'w') as f:
                f.write(model_id)
        except:
            pass
    
    def load_selected_model(self):
        try:
            path = os.path.join(self.data_path, 'model.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        except:
            pass
        return ""
    
    def save_game_to_history(self, game):
        history = self.load_history()
        history.append(game)
        history = history[-30:]
        try:
            path = os.path.join(self.data_path, 'history.json')
            with open(path, 'w') as f:
                json.dump(history, f)
        except:
            pass
    
    def load_history(self):
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

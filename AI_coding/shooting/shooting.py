import pygame
import random
import sys

# 初期化
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# 色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# プレイヤークラス
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 40))
        self.image.fill((0, 128, 255))
        self.rect = self.image.get_rect(center=(WIDTH//2, HEIGHT-50))
        self.speed = 5

    def update(self, keys):
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < WIDTH:
            self.rect.x += self.speed

# 弾クラス
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = -7

    def update(self):
        self.rect.y += self.speed
        if self.rect.bottom < 0:
            self.kill()

# 敵クラス
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, speed):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect(topleft=(x, y))
        self.speed = speed

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > HEIGHT:
            self.kill()

# ステージクリア表示
def stage_clear(screen, reward):
    text = font.render(f"Stage Clear! +{reward} Coins", True, WHITE)
    rect = text.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(text, rect)
    pygame.display.update()
    pygame.time.delay(2000)

# ゲームループ
def main():
    player = Player()
    player_group = pygame.sprite.Group(player)
    bullet_group = pygame.sprite.Group()
    enemy_group = pygame.sprite.Group()

    stage = 1
    reward = 0
    spawn_rate = 100  # 敵の出現間隔（小さいほど頻繁）

    running = True
    frame_count = 0

    while running:
        clock.tick(60)
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bullet = Bullet(player.rect.centerx, player.rect.top)
                bullet_group.add(bullet)

        # 更新
        player_group.update(keys)
        bullet_group.update()
        enemy_group.update()

        # 敵生成
        frame_count += 1
        if frame_count % spawn_rate == 0:
            x = random.randint(0, WIDTH - 30)
            speed = random.randint(1, 3) + stage // 2
            enemy = Enemy(x, 0, speed)
            enemy_group.add(enemy)

        # 衝突判定
        hits = pygame.sprite.groupcollide(enemy_group, bullet_group, True, True)

        # 画面描画
        screen.fill(BLACK)
        player_group.draw(screen)
        bullet_group.draw(screen)
        enemy_group.draw(screen)

        # リワード表示
        reward_text = font.render(f"Coins: {reward}", True, WHITE)
        screen.blit(reward_text, (10, 10))

        pygame.display.update()

        # ステージクリア判定
        if len(enemy_group) == 0 and frame_count > 300:
            stage_reward = stage * 10
            reward += stage_reward
            stage_clear(screen, stage_reward)
            stage += 1
            spawn_rate = max(10, spawn_rate - 2)  # ステージが進むと出現間隔短縮
            frame_count = 0

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

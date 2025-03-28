import random
import time
from blackjack import *

import pygame

#pygame init
pygame.init()

screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("BlackJack Game")

running = True

background_color = (34, 139, 34)
screen.fill(background_color)


cards_image = {}
for rank in RANKS:
    for suit in SUITS:
        filename = f"{rank}{suit}.png"
        cards_image[(rank, suit)] = pygame.image.load(f"img/{filename}")

#game setup
random.seed(time.time())
game = BlackJack()


def show_card_at(card, x, y):
    if card:
        card_image = cards_image.get((card[0], card[1]))
        if card_image:
            screen.blit(card_image, (x, y))

def display_all_hands(game):

    dealer_x = screen_width // 2 - len(game.dealer.hand) * 40 // 2
    dealer_y = 50
    for i, card in enumerate(game.dealer.hand):
        show_card_at(card, dealer_x + i * 40, dealer_y)

    # Display dealer's score below the cards
    dealer_score_font = pygame.font.Font(None, 36)
    dealer_score_text = dealer_score_font.render(f"Score: {game.dealer.get_score()}", True, (255, 255, 255))
    screen.blit(dealer_score_text, (dealer_x, dealer_y + 100))

    player1_x = 50
    player1_y = screen_height - 150
    for i, card in enumerate(game.players[0].hand):
        show_card_at(card, player1_x + i * 40, player1_y)

    # Display player 1's score above the cards
    player1_score_font = pygame.font.Font(None, 36)
    player1_score_text = player1_score_font.render(f"Score: {game.players[0].get_score()}", True, (255, 255, 255))
    screen.blit(player1_score_text, (player1_x, player1_y - 50))

    player2_x = screen_width - len(game.players[1].hand) * 40 - 50
    player2_y = screen_height - 150
    for i, card in enumerate(game.players[1].hand):
        show_card_at(card, player2_x + i * 40, player2_y)

    # Display player 2's score above the cards
    player2_score_font = pygame.font.Font(None, 36)
    player2_score_text = player2_score_font.render(f"Score: {game.players[1].get_score()}", True, (255, 255, 255))
    screen.blit(player2_score_text, (player2_x, player2_y - 50))


def create_buttons():
    button_width = 100
    button_height = 50
    hit_button_x = screen_width // 5 - button_width // 2
    hit_button_y = screen_height - 250
    stand_button_x = 2 * screen_width // 5 - button_width // 2
    stand_button_y = screen_height - 250

    hit_button_rect = pygame.Rect(hit_button_x, hit_button_y, button_width, button_height)
    stand_button_rect = pygame.Rect(stand_button_x, stand_button_y, button_width, button_height)

    pygame.draw.rect(screen, (0, 128, 255), hit_button_rect)
    hit_font = pygame.font.Font(None, 36)
    hit_text = hit_font.render("Hit", True, (255, 255, 255))
    screen.blit(hit_text, (hit_button_x + 25, hit_button_y + 12))

    pygame.draw.rect(screen, (255, 0, 0), stand_button_rect)
    stand_text = hit_font.render("Stand", True, (255, 255, 255))
    screen.blit(stand_text, (stand_button_x + 10, stand_button_y + 12))

    return hit_button_rect, stand_button_rect



clock = pygame.time.Clock()
FPS = 60

hit_button, stand_button = create_buttons()

while running:
    
    screen.fill(background_color)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Human player input handling
        if game.playerTurn == 0 and not game.players[0].finishedTurn:
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if hit_button.collidepoint(event.pos):
                    game.players[0].draw(game.deck)
                    print("hit")
                elif stand_button.collidepoint(event.pos):
                    game.players[0].stand()
                    print("stand")


    turnFinished = game.update()

    if game.playerTurn == 1:
        game.players[1].play(game.deck)

    if turnFinished:
        pass

    display_all_hands(game)

    pygame.display.flip()

    clock.tick(FPS)

pygame.quit()





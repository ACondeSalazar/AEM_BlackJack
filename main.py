import pygame
import sys
import time
from blackjack_deck import Deck, Hand
from constants import *

gamefont = textfont

white = light_slat
# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()

# Game Display Setup
gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Multiplayer Blackjack')
gameDisplay.fill(background_color)
pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))


# Render text on screen
def render_text(text, font, color, x, y):
    textSurface = font.render(text, True, color)
    textRect = textSurface.get_rect(center=(x, y))
    gameDisplay.blit(textSurface, textRect)
    pygame.display.update()


def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    
    pygame.draw.rect(gameDisplay, ac if x + w > mouse[0] > x and y + h > mouse[1] > y else ic, (x, y, w, h))
    
    if x + w > mouse[0] > x and y + h > mouse[1] > y and click[0] == 1 and action:
        action()

    render_text(msg, font, black, x + (w/2), y + (h/2))

class BlackjackGame:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.deck = Deck()
        self.dealer = Hand()
        self.players = [Hand() for _ in range(num_players)]
        self.current_player = 0
        self.deck.shuffle()

    def deal(self):
        # Deal initial cards
        for i in range(2):
            self.dealer.add_card(self.deck.deal())
            for player in self.players:
                player.add_card(self.deck.deal())

        self.display_hands()
        self.blackjack_check()

    def display_hands(self):
        # Show dealer’s hand
        render_text("Dealer's Hand:", gamefont, white, 500, 150)
        gameDisplay.blit(pygame.image.load(f'img/{self.dealer.card_img[0]}.png'), (400, 200))
        gameDisplay.blit(pygame.image.load('img/back.png'), (550, 200))

        # Show each player's hand
        y_offset = 400
        for i, player in enumerate(self.players):
            render_text(f"Player {i + 1} Hand:", gamefont, white, 500, y_offset - 50)
            for j, card in enumerate(player.card_img):
                gameDisplay.blit(pygame.image.load(f'img/{card}.png'), (300 + (j * 110), y_offset))
            y_offset += 120

        pygame.display.update()

    def blackjack_check(self):
        # Check for instant Blackjacks
        if self.dealer.value == 21:
            render_text("Dealer has Blackjack!", game_end, red, 500, 250)
            time.sleep(3)
            self.finish_game()
        elif any(player.value == 21 for player in self.players):
            for i, player in enumerate(self.players):
                if player.value == 21:
                    render_text(f"Player {i + 1} has Blackjack!", game_end, green, 500, 250)
                    time.sleep(3)
            self.finish_game()

    def hit(self):
        current_player = self.players[self.current_player]
        current_player.add_card(self.deck.deal())
        self.display_hands()

        if current_player.value > 21:
            render_text(f"Player {self.current_player + 1} Busts!", game_end, red, 500, 250)
            time.sleep(2)
            self.next_turn()

    def stand(self):
        render_text(f"Player {self.current_player + 1} Stands!", gamefont, white, 500, 250)
        time.sleep(1)
        self.next_turn()

    def next_turn(self):
        self.current_player += 1
        if self.current_player >= len(self.players):  
            self.dealer_turn()
        else:
            render_text(f"Player {self.current_player + 1}'s Turn", gamefont, white, 500, 80)

    def dealer_turn(self):
        render_text("Dealer's Turn", gamefont, white, 500, 80)
        time.sleep(2)
        gameDisplay.blit(pygame.image.load(f'img/{self.dealer.card_img[1]}.png'), (550, 200))
        pygame.display.update()
        time.sleep(2)

        while self.dealer.value < 17:
            self.dealer.add_card(self.deck.deal())
            self.display_hands()
            time.sleep(1)

        self.check_results()

    def check_results(self):
        for i, player in enumerate(self.players):
            if player.value > 21:
                render_text(f"Player {i + 1} Busts!", game_end, red, 500, 250)
            elif self.dealer.value > 21 or player.value > self.dealer.value:
                render_text(f"Player {i + 1} Wins!", game_end, green, 500, 250)
            elif player.value < self.dealer.value:
                render_text(f"Player {i + 1} Loses!", game_end, red, 500, 250)
            else:
                render_text(f"Player {i + 1} Ties!", game_end, grey, 500, 250)
            time.sleep(2)

        self.finish_game()

    def finish_game(self):
        render_text("Play again? Press Deal!", gamefont, white, 200, 80)
        time.sleep(3)
        self.__init__(self.num_players)  # Reset game
        gameDisplay.fill(background_color)
        pygame.draw.rect(gameDisplay, grey, pygame.Rect(0, 0, 250, 700))
        pygame.display.update()

    def exit_game(self):
        pygame.quit()
        sys.exit()


blackjack = BlackjackGame(num_players=3)  # Change the number of players here!

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    button("Deal", 30, 100, 150, 50, light_slat, dark_slat, blackjack.deal)
    button("Hit", 30, 200, 150, 50, light_slat, dark_slat, blackjack.hit)
    button("Stand", 30, 300, 150, 50, light_slat, dark_slat, blackjack.stand)
    button("EXIT", 30, 500, 150, 50, light_slat, dark_red, blackjack.exit_game)

    pygame.display.flip()

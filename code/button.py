import pygame

class Button:
    """
    Button for the pygame interface.
    """

    RECT_PADDING = 10
    RECT_BORDER_RADIUS = 4

    def __init__(
        self,
        x: int,
        y: int,
        text: str,
        canvas: pygame.Surface,
        font: pygame.font.Font,
        textColor: tuple[int, int, int],
        bgColor: tuple[int, int, int],
        action=None,
        width=None,
        height=None,
    ):
        self.text = font.render(text, True, textColor)
        self.canvas = canvas
        self.textRect = self.text.get_rect(center=(x, y))
        self.bgColor = bgColor

        if width is None:
            width = self.textRect.width

        if height is None:
            height = self.textRect.height

        paddingWidth = width - self.textRect.width + self.RECT_PADDING
        paddingHeight = height - self.textRect.height + self.RECT_PADDING

        self.paddingRect = self.textRect.inflate(paddingWidth, paddingHeight)
        self.action = action

    def draw(self):
        if self.bgColor is not None:
            pygame.draw.rect(
                self.canvas, self.bgColor, self.paddingRect, 0, self.RECT_BORDER_RADIUS
            )
        self.canvas.blit(self.text, self.textRect)

    def mouse_collision(self, x, y):
        return self.paddingRect.collidepoint(x, y)


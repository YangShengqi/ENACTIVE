import pygame
import time


class Stick:
    def __init__(self):
        pygame.joystick.init()
        pygame.display.init()
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        print("stick num:", len(self.joysticks))
        self.shoot_pres = [0]*len(self.joysticks)
        self.target_pres = [0]*len(self.joysticks)
        self.stick_num = len(self.joysticks)

    def sample(self):
        pygame.event.pump()
        #pygame.event.clear()
        actions = []
        for i in range(self.stick_num):
            self.joysticks[i].init()
            axis0 = self.joysticks[i].get_axis(0)
            axis1 = self.joysticks[i].get_axis(1)
            shoot = self.joysticks[i].get_button(1)
            target = self.joysticks[i].get_button(0)
            if shoot == 0 and self.shoot_pres[i] == 1:
                shoot = 1
                self.shoot_pres[i] = 0
            else:
                self.shoot_pres[i] = shoot
                shoot = 0
            if target == 0 and self.target_pres[i] == 1:
                target = 1
                self.target_pres[i] = 0
            else:
                self.target_pres[i] = target
                target = 0
            actions.append({"axis0":axis0,"axis1":axis1,"shoot":shoot,"target":target})
        return actions


if __name__ == "__main__":
    s = Stick()
    while True:
        time.sleep(0.5)
        s.sample()

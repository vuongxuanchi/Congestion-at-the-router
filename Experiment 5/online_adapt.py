#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buoc 2 - TU DIEU CHINH KHI CHAY (online adaptation) cho DN-dCoD.

Y tuong: chinh sach da huan luyen (o 10 Mbps) KHONG bi dong bang; khi chay tren vet moi no
tiep tuc hoc nhe bang chinh phan thuong cua moi truong:
  - epsilon rat nho (mac dinh 0.01), toc do hoc nho (1e-4, thap hon luc huan luyen 1e-3),
    cap nhat 1 lan moi 4 goi, bo nho kinh nghiem CHI chua kinh nghiem moi cua lan chay nay.
  - PHUONG AN DU PHONG (rollback): chia dong goi thanh cua so 5000 goi; giu lai bo trong so
    cho phan thuong TB cua so tot nhat tu truoc toi nay (ban dau = chinh sach dong bang).
    Neu phan thuong cua so hien tai TUT qua `margin` (ty le) so voi cua so tot nhat -> khoi phuc
    trong so tot nhat, xoa bo nho, va dem 1 lan rollback.
  Luu y: cac cua so la cac doan vet khac nhau (tai khong dong deu) nen so sanh nay chi la LUOI AN TOAN
  thuan thuc, khong phai kiem dinh thong ke; so lan rollback duoc in ra de doc ket qua.

Dung qua eval_da_tai.py:  --methods dn_dcod_khong_nhan_online dn_dcod_online
"""
import copy
import random
import numpy as np


class OnlineAgent:
    """Boc 1 DQNAgent (train_dqn_dcod.DQNAgent da nap trong so). Tuong thich env.run_greedy_and_export:
    select_action(s, greedy) + attach(env) de nhan (reward, s') sau moi buoc."""

    def __init__(self, base, eps=0.01, lr=1e-4, update_every=4, buffer_size=20000,
                 window=5000, margin=0.25, rollback=True):
        import torch
        import train_dqn_dcod as tdd
        self.torch = torch
        self.base = base
        self.eps, self.update_every = float(eps), int(update_every)
        self.window, self.margin, self.rollback = int(window), float(margin), bool(rollback)
        base.q_net.train()
        base.opt = torch.optim.Adam(base.q_net.parameters(), lr=lr)
        base.buffer = tdd.ReplayBuffer(buffer_size)
        base.global_step = 0
        base.target_net.load_state_dict(base.q_net.state_dict())
        self.best_state = copy.deepcopy(base.q_net.state_dict())   # = chinh sach dong bang
        self.best_R = None
        self._cand = copy.deepcopy(base.q_net.state_dict())
        self._win_r, self._win_n = 0.0, 0
        self.n_steps = self.n_updates = self.n_rollback = self.n_random = 0
        self._s = self._a = None

    # --- chon hanh dong ---
    def select_action(self, s, greedy=True):
        self._s = np.asarray(s, dtype=np.float32)
        if random.random() < self.eps:
            self.n_random += 1
            self._a = random.randrange(self.base.action_dim)
        else:
            self._a = self.base.select_action(s, greedy=True)
        return self._a

    # --- nhan phan thuong sau buoc ---
    def attach(self, env):
        orig = env.step
        agent = self

        def step(action):
            s2, r, done, info = orig(action)
            agent.observe(r, s2, done)
            return s2, r, done, info
        env.step = step
        return env

    def observe(self, r, s2, done):
        self.n_steps += 1
        self.base.store(self._s, self._a, float(r), np.asarray(s2, dtype=np.float32), done)
        if self.n_steps % self.update_every == 0:
            if self.base.train_step() is not None:
                self.n_updates += 1
        self._win_r += float(r)
        self._win_n += 1
        if self._win_n >= self.window:
            self._end_window()

    def _end_window(self):
        R = self._win_r / self._win_n
        if self.best_R is None or R > self.best_R:
            self.best_R, self.best_state = R, self._cand      # cua so tot nhat: trong so da sinh ra no
        elif self.rollback and R < self.best_R - self.margin * max(abs(self.best_R), 1e-3):
            self.base.q_net.load_state_dict(self.best_state)
            self.base.target_net.load_state_dict(self.best_state)
            self.base.buffer.buf.clear()
            self.n_rollback += 1
        self._cand = copy.deepcopy(self.base.q_net.state_dict())
        self._win_r, self._win_n = 0.0, 0

    def summary(self):
        return (f"[online] {self.n_steps} goi | {self.n_updates} lan cap nhat mang | "
                f"{self.n_random} hanh dong ngau nhien ({100 * self.n_random / max(self.n_steps, 1):.2f}%) | "
                f"{self.n_rollback} lan rollback | R tot nhat={self.best_R if self.best_R is None else round(self.best_R, 4)}")

#CLI Forense - Image Toolkit
#Autori: Cristiano Pistorio, Sofia Manno - 2025
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from forensic_cli.session import Session

Action = Callable[[Session], None]


@dataclass
class MenuItem:
    key: str
    label: str
    action: Optional[Action] = None
    submenu: Optional["Menu"] = None


class Menu:
    def __init__(self, title: str):
        self.title = title
        self.items: List[MenuItem] = []

    def add(self, key: str, label: str, action: Optional[Action] = None, submenu: Optional["Menu"] = None):
        self.items.append(MenuItem(key=key, label=label, action=action, submenu=submenu))

    def run(self, session: Session):
        while True:
            print("\n" + "=" * 70)
            print(self.title)
            print("=" * 70)
            for item in self.items:
                print(f"[{item.key}] {item.label}")
            choice = input("Seleziona un'opzione: ").strip().lower()
            matched = next((it for it in self.items if it.key.lower() == choice), None)
            if not matched:
                print("Opzione non valida. Riprova.")
                continue
            if matched.submenu is not None:
                matched.submenu.run(session)
            elif matched.action is not None:
                matched.action(session)
            else:
                return


__all__ = ["Menu", "MenuItem", "Action"]


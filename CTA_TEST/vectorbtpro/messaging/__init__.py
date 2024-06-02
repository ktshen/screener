# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules for messaging."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.messaging.telegram import *

__import_if_installed__ = dict()
__import_if_installed__["telegram"] = "telegram"

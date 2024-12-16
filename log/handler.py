import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<k>{time:YYYY-MM-DD HH:mm:ss.SSS}</k> | <level>{level: <6}</level> | N=<c>{name}</c>:M=<c>{module}</c>:F=<c>{function}</c>:L=<c>L{line}</c> - <level>{message}</level>", colorize=True)
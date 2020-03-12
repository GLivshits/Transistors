import telegram
import socks
import socket

socks.set_default_proxy(proxy_type=socks.PROXY_TYPE_SOCKS5, addr='144.76.99.207', port=83)

socket.socket = socks.socksocket

bot = telegram.Bot(token='1069929691:AAErLtdcVDbP3PBriZj-tyAiYnunsxbEcvQ')

bot.send_message(text = 'Hello!', chat_id = 213049562, timeout=5)
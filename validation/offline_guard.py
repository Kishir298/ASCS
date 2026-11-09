"""Test harness: prohibit external DNS and socket connections before test imports."""
import socket
import ipaddress
_real_getaddrinfo = socket.getaddrinfo
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def allowed(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def lookup(host, *args, **kwargs):
    if not allowed(host):
        raise OSError('External DNS disabled by offline test harness')
    return _real_getaddrinfo(host, *args, **kwargs)


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not allowed(address[0]):
        raise OSError('External connection disabled by offline test harness')
    return _real_connect(self, address)


def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not allowed(address[0]):
        raise OSError('External connection disabled by offline test harness')
    return _real_connect_ex(self, address)

socket.getaddrinfo = lookup
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex

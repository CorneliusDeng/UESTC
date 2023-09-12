section .text

global main

main:

mov eax, 1 	;write syscall number
mov rdi, 1 	;fd
mov rsi, msge	;buffer
mov rdx, 14	;length
syscall

mov eax, 60	;exit syscall number
mov rdi, 5      ;exit code
syscall

msge:
db "Hello World!", 0ah, 0dh

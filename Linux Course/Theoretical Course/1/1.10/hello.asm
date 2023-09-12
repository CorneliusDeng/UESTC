section .text

global main

main:

mov eax, 4 	;write syscall number
mov ebx, 1 	;fd
mov ecx, msge	;buffer
mov edx, 14	;length
int 80h

mov eax, 1	;exit syscall number
mov ebx, 5      ;exit code
int 80h

msge:
db "Hello World!", 0ah, 0dh

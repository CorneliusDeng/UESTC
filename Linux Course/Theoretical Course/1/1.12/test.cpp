int main()
{
    long ret = 0;
    int fd = 0;
    const char *buffer = "Hello World!\r\n";
    unsigned int size = 14;

    asm volatile("movl $4, %%eax \n\t"
	    		  "movl %1, %%ebx \n\t"
			  "movq %2, %%rcx \n\t"
			  "movq %3, %%rdx \n\t"
			  "int $0x80	 \n\t"
			  "movq %%rax, %0 \n\t" : 
			  "=m"(ret) : "m"(fd), "m"(buffer), "m"(size) : 
			  "%rax", "%ebx", "%rcx", "%rdx", "memory");

    return 0;
}

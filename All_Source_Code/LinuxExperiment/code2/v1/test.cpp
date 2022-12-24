#include "Temp.h"

int main(){
    
    	Temp a(12);
    	a.Serialize("data.txt");
		Temp b;
		b.Deserialize("data.txt");
    	b.f();
}

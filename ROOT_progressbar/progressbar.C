#include <iostream>
#include "./progressbar.h"

void progressbar() {
  ProgressBar bar1; 
    for (int i = 0; i < 100.; ++i) {    
        if (i/10 == TRUE) bar1.progress();
    }       
}

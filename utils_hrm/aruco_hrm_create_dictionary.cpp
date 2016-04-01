/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/

#include "highlyreliablemarkers.h"
#include <iostream>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;

typedef std::vector< int > Word;




class MarkerGenerator {

  private:
    int _nTransitions;
    std::vector< int > _transitionsWeigth;
    int _totalWeigth;
    int _n;

  public:
    MarkerGenerator(int n) {
        _n = n;
        _nTransitions = n - 1;
        _transitionsWeigth.resize(_nTransitions);
        _totalWeigth = 0;
        for (int i = 0; i < _nTransitions; i++) {
            _transitionsWeigth[i] = i;
            _totalWeigth += i;
        }
    }

    aruco::MarkerCode generateMarker() {

        aruco::MarkerCode emptyMarker(_n);

        for (int w = 0; w < _n; w++) {
            Word currentWord(_n, 0);
            int randomNum = rand() % _totalWeigth;
            int currentNTransitions = _nTransitions - 1;
            for (int k = 0; k < _nTransitions; k++) {
                if (_transitionsWeigth[k] > randomNum) {
                    currentNTransitions = k;
                    break;
                }
            }
            std::vector< int > transitionsIndexes(_nTransitions);
            for (int i = 0; i < _nTransitions; i++)
                transitionsIndexes[i] = i;
            std::random_shuffle(transitionsIndexes.begin(), transitionsIndexes.end());

            std::vector< int > selectedIndexes;
            for (int k = 0; k < currentNTransitions; k++)
                selectedIndexes.push_back(transitionsIndexes[k]);
            std::sort(selectedIndexes.begin(), selectedIndexes.end());
            int currBit = rand() % 2;
            int currSelectedIndexesIdx = 0;
            for (int k = 0; k < _n; k++) {
                currentWord[k] = currBit;
                if (currSelectedIndexesIdx < selectedIndexes.size() && k == selectedIndexes[currSelectedIndexesIdx]) {
                    currBit = 1 - currBit;
                    currSelectedIndexesIdx++;
                }
            }

            for (int k = 0; k < _n; k++)
                emptyMarker.set(w * _n + k, bool(currentWord[k]), false);
        }

        return emptyMarker;
    }
};




int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "Invalid number of arguments" << endl;
        cerr << "Usage: outputfile.yml dictSize n  \n \
      outputfile.yml: output file for the dictionary \n \
      dictSize: number of markers to add to the dictionary \n \
      n: marker size." << endl;
        exit(-1);
    }

    aruco::Dictionary D;
    unsigned int dictSize = atoi(argv[2]);
    unsigned int n = atoi(argv[3]);

    unsigned int tau = 2 * ((4 * ((n * n) / 4)) / 3);
    std::cout << "Tau: " << tau << std::endl;

    srand(time(NULL));

    MarkerGenerator MG(n);

    const int MAX_UNPRODUCTIVE_ITERATIONS = 100000;
    int currentMaxUnproductiveIterations = MAX_UNPRODUCTIVE_ITERATIONS;

    unsigned int countUnproductive = 0;
    while (D.size() < dictSize) {

        aruco::MarkerCode candidate;
        candidate = MG.generateMarker();

        if (candidate.selfDistance() >= tau && D.distance(candidate) >= tau) {
            D.push_back(candidate);
            std::cout << "Accepted Marker " << D.size() << "/" << dictSize << std::endl;
            countUnproductive = 0;
        } else {
            countUnproductive++;
            if (countUnproductive == currentMaxUnproductiveIterations) {
                tau--;
                countUnproductive = 0;
                std::cout << "Reducing Tau to: " << tau << std::endl;
                if (tau == 0) {
                    std::cerr << "Error: Tau=0. Small marker size for too high number of markers. Stop" << std::endl;
                    break;
                }
                if (D.size() >= 2)
                    currentMaxUnproductiveIterations = MAX_UNPRODUCTIVE_ITERATIONS;
                else
                    currentMaxUnproductiveIterations = MAX_UNPRODUCTIVE_ITERATIONS / 15;
            }
        }
    }

    D.tau0 = tau;
    D.toFile(argv[1]);
}

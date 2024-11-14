/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <stdio.h>
#include <tvm_runtime.h>
#include <tvmgen_picodet.h>

#include "uart.h"

// Header files generated by convert_image.py
#include "inputs.h"
#include "outputs.h"

int main(int argc, char **argv) {
  uart_init();
  printf("Starting PicoDet inference:\n");
  struct tvmgen_picodet_outputs rec_outputs = {
      .output0 = output0, .output1 = output1,
  };
  struct tvmgen_picodet_inputs rec_inputs = {
      .image = input,
  };

  tvmgen_picodet_run(&rec_inputs, &rec_outputs);

  // post process
  for (int i = 0; i < output0_len / 4; i++) {
    float score = 0;
    int32_t class = 0;
    for (int j = 0; j < 80; j++) {
      if (output1[i + j * 2125] > score) {
        score = output1[i + j * 2125];
        class = j;
      }
    }
    if (score > 0.1 && output0[i * 4] > 0 && output0[i * 4 + 1] > 0) {
      printf("box: %f, %f, %f, %f, class: %d, score: %f\n", output0[i * 4] * 2,
             output0[i * 4 + 1] * 2, output0[i * 4 + 2] * 2,
             output0[i * 4 + 3] * 2, class, score);
    }
  }
  return 0;
}

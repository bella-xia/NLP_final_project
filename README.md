# NLP_final_project


What we have tried:

1. Five FAILED and WRONG fine-tuning scheme (thank you Jiefu :)

2. A dataset with unspecific outputs that is very hard to derive its instructions (ANOTHER FAILURE)

3. The Alpaca dataset, filtered with no input, forward model fine tuning. Some success, with specific choice of temperature (~1.5) and top-k filtering (30 / 100 or something in between), the result looks descent. But a general problem is that 
        (i) the generation is not very fluent; 
        (ii) generation can degrade the further away from the instruction, which makes any attempt to backward predict a little bit troublesome

4. backward fine tuning but CALIBERATED (under the suggestion of YOU KNOW WHO) (GIGANTIC FAILURE)

Plan to move forward:

1. use backward model to do backward generation WITH TRUE LABEL UNCALIBRATED

2. use smart Tianjian's suggestions

    - Use a forward model train on both forward and backward generation

      (i) doesn't change positional embedding --> model needs to learn both the natural language and the UNNAURAL Peter West language

      (ii) changes positinal embedding --> model kind of like Bert learning to predict bi-directionally
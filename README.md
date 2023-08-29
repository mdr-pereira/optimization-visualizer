# optimization-visualizer

## General Idea

Create a visualizer for optimization issues / optimizers.

We want to be able to visualize the landscape of any function w/in given bounds (otherwise it's somewhat unsolvable), and in 3D cases this can be done through a point mesh taken at arbitrary intervals. The main focus of this project
would be to be able to visualize the action of modern optimizers (whether 2D or 3D, whether gradient descent or more simple optimizers), and see the path they take.

In terms of importance of this project I think it could be a great tool to help freshmen understand how the optimization of functions occurs, and better yet, to help us seniors to visualize the difference between multiple optimizers,
s/as ADAM, SGD, SGD w/ momentum.

## Implementation

Start off w/ simple implementation for 2D optimization issues, with a more simplistic optimizer, such as simulated annealing (or smth along these lines).
-> Implementation should be started w/ python for simplicity, but then can be changed over to more potent 3D engines.

Now, we have 2 routes of improvement: Generalization, Complexity.
- Generalization -> implement generalistic function input into program, ie. make it possible to use any landscape, and implement more (or arbitrary) optimizers.
- Complexity -> implement same functionalities as we have at this point to 3-dim problems (max due to visualization issues, we could take n-dim issues, but visualization would be done over PCA), this requires (in my opinion) a more
		complex visual engine than python (TBD).

We could even separate ourselves into 2 teams, one tries the generalization route, while another tries the other route.

 

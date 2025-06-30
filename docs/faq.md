# Common Questions

Here are some answers to the questions we get most often

**What settings do you recommend?**

The current threshold defaults in all interfaces are what we found works best, in our own testing. We recommend
starting there and making small adjustments as needed. It may be useful to process just a subset of your images for
testing purposes for the sake of quickly iterating and finding what works best for you.

We typically use a pixel buffer of 2 to slightly expand our masked areas.

**What number of workers should I use?**

We typically use the default of 4. If you have a really powerful machine with a lot of memory, you might benefit from
increasing this value. If you're machine is grinding to a halt, lower it.

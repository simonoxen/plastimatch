-- James Shackleford
-- Feb. 19th, 2012
-- Example of LUA scripted registration
--
--   This is still under active development.  Not
--   all registration parameters have been exposed
--   to the LUA interface, so not all features are
--   available.  Your millage may vary.
--
-- Install LUA 5.1 and build with PLM_CONFIG_ENABLE_LUA = ON
-- invoke this script with: ./plastimatch script example.lua
--

io.write ("Hi!  I'm a LUA script.  I'm going to be making your\n")
io.write ("life much better by (eventually) allowing you to pipline\n")
io.write ("everything plastimatch can do... not just registration!\n")
io.write ("Let's get started!\n\n")


-- Registration Parameters
regp = {
    fixed = "/home/tshack/data/reg/set01/hi_gcs.mha",
    moving = "/home/tshack/data/reg/set01/synth_radial_img.mha",
    vf_out = "/home/tshack/lua_test/vf.mha",
    img_out = "/home/tshack/lua_test/warp.mha"
}

stage_1 = {
    xform = "bspline",
    metric = "mse",
    optim = "lbfgsb",
    impl = "plastimatch",
    threading = "openmp",
    iterations = 21
}

stage_1 = {
    xform = "bspline",
    metric = "mse",
    optim = "lbfgsb",
    impl = "plastimatch",
    threading = "openmp",
    iterations = 21
}

stage_2 = {
    xform = "bspline",
    metric = "mse",
    optim = "lbfgsb",
    impl = "plastimatch",
    threading = "openmp",
    iterations = 40
}

-- Notice how call stack is pascal style,
-- C receives &regp @ the top of the stack
-- This will probably be reversed in the future
-- so that it is less confusing for the user,
-- but it is easier for the developer; so it stays
-- for now ;-)
a = register(stage_2, stage_1, regp)


io.write ("Return value (", a, ").\n")
io.write ("See, now.  Wasn't that swell?\n")

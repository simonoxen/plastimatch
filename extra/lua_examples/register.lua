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


-- You can write to the screen with io.write()
io.write ("Hi!  I'm a LUA script.  I'm going to be making your\n")
io.write ("life much better by (eventually) allowing you to pipline\n")
io.write ("everything plastimatch can do... not just registration!\n")
io.write ("Let's get started!\n\n")


-- Registration Parameters
global = {
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
    max_its = 10
}

stage_2 = {
    xform = "bspline",
    metric = "mse",
    optim = "lbfgsb",
    impl = "plastimatch",
    threading = "openmp",
    grid_spac = {20.0, 20.0, 20.0},
    max_its = 21
}

stage_3 = {
    xform = "bspline",
    metric = "mse",
    optim = "lbfgsb",
    impl = "plastimatch",
    threading = "openmp",
    grid_spac = {10.0, 10.0, 10.0},
    max_its = 40
}

-- NOTE: register() is VARIADIC, so you can keep adding stages
a = register(stage_3, stage_2, stage_1, global)


-- Display return values with io.write() like this
io.write ("Return value (", a, ").\n")
io.write ("See, now.  Wasn't that swell?\n")

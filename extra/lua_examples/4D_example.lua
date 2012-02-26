-- James Shackleford
-- Feb. 25th, 2012
-- Example of LUA scripted 4D/batch registration
--
--   The LUA scripting interface is in active development.
--   Not all registration parameters have been exposed to
--   the LUA interface, so not all features are available.
--   Your millage may vary, but will probably be pretty
--   good.
--
-- Install LUA 5.1 and build with PLM_CONFIG_ENABLE_LUA = ON
-- invoke this script with: plastimatch script 4D_example.lua
--
--------------------------------------------------------------

-- Okay, let's get started.

------------------------------------------------------
-- First, let's define were our input and output
-- files are going to live.
------------------------------------------------------

-- We can save ourselves some typing and create fewer
-- opportunities to makes mistakes by using .. to
-- glue strings together
phase_dir = "/home/tshack/data/reg/0133/"
vf_dir    = phase_dir .. "vf/"
warp_dir  = phase_dir .. "warp/"

-- you can define correlated inputs and outputs using a table like this one
input_phases = {
    { moving = "0010.mha", vf_out = "0010_vf.mha", img_out = "0010_warp.mha" },
    { moving = "0011.mha", vf_out = "0011_vf.mha", img_out = "0011_warp.mha" },
    { moving = "0012.mha", vf_out = "0012_vf.mha", img_out = "0012_warp.mha" },
    { moving = "0013.mha", vf_out = "0013_vf.mha", img_out = "0013_warp.mha" },
    { moving = "0014.mha", vf_out = "0014_vf.mha", img_out = "0014_warp.mha" },
    { moving = "0016.mha", vf_out = "0016_vf.mha", img_out = "0016_warp.mha" },
    { moving = "0017.mha", vf_out = "0017_vf.mha", img_out = "0017_warp.mha" },
    { moving = "0018.mha", vf_out = "0018_vf.mha", img_out = "0018_warp.mha" },
    { moving = "0019.mha", vf_out = "0019_vf.mha", img_out = "0019_warp.mha" }
}


------------------------------------------------------
-- Now we can define some registration parameters.
-- The format here is very similar to the traditional
-- Plastimatch command file
------------------------------------------------------
global = {
    fixed = phase_dir .. "0015.mha",
    moving  = "",  -- we will fill these in later for each
    vf_out  = "",  -- phase by cycling through the input_phases
    img_out = ""   -- table we created above
}

-- define some registration stages
stage_1 = {
    xform = "bspline",
    metric = "mse",
    optim = "lbfgsb",
    impl = "plastimatch",
    threading = "openmp",
    max_its = 20,
    grid_spac = {30, 30, 30},
    res = {4, 4, 1}
}

stage_2 = {
    grid_spac = {20, 20, 20},
    res = {2, 2, 1}
}

stage_3 = {
    grid_spac = {15, 15, 15},
    res = {2, 2, 1}
}

stage_4 = {
    grid_spac = {10, 10, 10},
    res = {1, 1, 1}
}


------------------------------------------------------
-- Now we will cycle through our input_phases table
-- one phase at a time.  For each phase we will:
--   1) Populate the moving, vf_out, and img_out
--      fields in the global registration settings.
--      This will tell Plastimatch where the input
--      and output files live.
--
--   2) Give ourselves some feedback by printing
--      the current phase under registration to
--      the screen using a banner.
--
--   3) Tell plastimatch to perform the registration
--      by using the register() command.
------------------------------------------------------
for i, phase in pairs (input_phases)
do
    -- concatenate strings with ..
    global.moving  = phase_dir .. phase.moving
    global.vf_out  = vf_dir .. phase.vf_out
    global.img_out = warp_dir .. phase.img_out

    -- give ourselves some feedback
    print ("===================================")
    print ("Registering phase " .. phase.moving)
    print ("===================================")

    -- call plastimatch to register this phase
    --
    -- note #1: stages are passed in *reverse* order and global
    --          is passed at the end.  stage_1 will be used 1st
    --          and stage_4 will be used last.
    --
    -- note #2: subsequent stages inherit previous stage parameters
    --          so, stage_2 inherits stage_1's parameters and then
    --          overwrites grid_spac and res.  This saves you some 
    --          typing.
    --
    -- note #3: there is no limit to the number of stages you can
    --          specify. you can even specify the same stage multiple
    --          times.
    register (stage_4, stage_3, stage_2, stage_1, global)
end

print ("Finished registering all phases!")

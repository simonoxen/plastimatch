-- James Shackleford
-- Feb. 29th, 2012
-------------------------
-- Examples of classes and their functionality
-- as I develop them.  This serves as both
-- documentation and a testbed.
--------------------------------------------------

-- Load an image
my_image = image.load ("test.mha")

-- "Save" (Write it to disk)
my_image:save ()

-- "Save-As" (Write it to disk as a new file)
my_image:save ("test-clone.mha")


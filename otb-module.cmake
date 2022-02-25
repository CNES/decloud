set(DOCUMENTATION "Remote clouds in optical imagery")

# define the dependencies of the include module and the tests
otb_module(OTBDecloud
  DEPENDS
    OTBTensorflow
  TEST_DEPENDS
    OTBTestKernel
    OTBCommandLine
  DESCRIPTION
    "${DOCUMENTATION}"
)

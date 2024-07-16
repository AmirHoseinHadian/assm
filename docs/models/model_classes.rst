.. currentmodule:: assm

Model classes
=============

These classes can be used to define different available models. Currently, 3 classes of models are implemented in `assm`:

1. Attentional drift diffusion models: :ref:`aDDModel <aDDModel>`
2. Gaze weighted linear accumulator models: :ref:`GLAModel_2A <GLAModel_2A>`, :ref:`GLAModel_nA <GLAModel_nA>`
3. Gaze advantage race diffusion models: :ref:`GARDModel_2A <GARDModel_2A>`, :ref:`GARDModel_nA <GARDModel_nA>`

All classes have a hierarchical and non-hierarchical version, and come with additional cognitive mechanisms that can be added or excluded.

Attentional drift diffusion models
-----------------------------------------------

.. _aDDModel:
.. autoclass:: assm.model.models_aDDM.aDDModel
    :members:

    .. automethod:: __init__

Gaze weighted linear accumulator models
------------------------------------------------

.. _GLAModel_2A:
.. autoclass:: assm.model.models_GLAM.GLAModel_2A
    :members:

    .. automethod:: __init__

.. _GLAModel_nA:
.. autoclass:: assm.model.models_GLAM.GLAModel_nA
    :members:

    .. automethod:: __init__

Gaze advantage race diffusion models
------------------------------------------------

.. _GARDModel_2A:
.. autoclass:: assm.model.models_GARD.GARDModel_2A
    :members:

    .. automethod:: __init__

.. _GARDModel_nA:
.. autoclass:: assm.model.models_GARD.GARDModel_nA
    :members:

    .. automethod:: __init__